; RUN: llc -mtriple=i386-apple-darwin %s -o %t
; RUN:  FileCheck %s < %t
; RUN:  FileCheck %s -check-prefix=GOT-EQUIV < %t

; GOT equivalent globals references can be replaced by the GOT entry of the
; final symbol instead.

%struct.data = type { i32, %struct.anon }
%struct.anon = type { i32, i32 }

; Check that these got equivalent symbols are never emitted or used
; GOT-EQUIV-NOT: _localgotequiv
; GOT-EQUIV-NOT: _extgotequiv
@localfoo = global i32 42
@localgotequiv = private unnamed_addr constant i32* @localfoo

@extfoo = external global i32
@extgotequiv = private unnamed_addr constant i32* @extfoo

; Don't replace GOT equivalent usage within instructions and emit the GOT
; equivalent since it can't be replaced by the GOT entry. @bargotequiv is
; used by an instruction inside @t0.
;
; CHECK: l_bargotequiv:
; CHECK-NEXT:  .long   _extbar
@extbar = external global i32
@bargotequiv = private unnamed_addr constant i32* @extbar

@table = global [4 x %struct.data] [
; CHECK-LABEL: _table
  %struct.data { i32 1, %struct.anon { i32 2, i32 3 } },
; Test GOT equivalent usage inside nested constant arrays.
; CHECK: .long   5
; CHECK-NOT: l_localgotequiv-(_table+20)
; CHECK-NEXT: L_localfoo$non_lazy_ptr-(_table+20)
  %struct.data { i32 4, %struct.anon { i32 5,
    i32 sub (i32 ptrtoint (i32** @localgotequiv to i32),
             i32 ptrtoint (i32* getelementptr inbounds ([4 x %struct.data], [4 x %struct.data]* @table, i32 0, i32 1, i32 1, i32 1) to i32))}
  },
; CHECK: .long   5
; CHECK-NOT: l_extgotequiv-(_table+32)
; CHECK-NEXT: L_extfoo$non_lazy_ptr-(_table+32)
  %struct.data { i32 4, %struct.anon { i32 5,
    i32 sub (i32 ptrtoint (i32** @extgotequiv to i32),
             i32 ptrtoint (i32* getelementptr inbounds ([4 x %struct.data], [4 x %struct.data]* @table, i32 0, i32 2, i32 1, i32 1) to i32))}
  },
; Test support for arbitrary constants into the GOTPCREL offset
; CHECK: .long   5
; CHECK-NOT: (l_extgotequiv-(_table+44))+24
; CHECK-NEXT: L_extfoo$non_lazy_ptr-(_table+20)
  %struct.data { i32 4, %struct.anon { i32 5,
    i32 add (i32 sub (i32 ptrtoint (i32** @extgotequiv to i32),
                      i32 ptrtoint (i32* getelementptr inbounds ([4 x %struct.data], [4 x %struct.data]* @table, i32 0, i32 3, i32 1, i32 1) to i32)),
                      i32 24)}
  }
], align 16

; Test multiple uses of GOT equivalents.
; CHECK-LABEL: _delta
; CHECK: .long   L_extfoo$non_lazy_ptr-_delta
@delta = global i32 sub (i32 ptrtoint (i32** @extgotequiv to i32),
                         i32 ptrtoint (i32* @delta to i32))

; CHECK-LABEL: _deltaplus:
; CHECK: .long   L_localfoo$non_lazy_ptr-(_deltaplus-55)
@deltaplus = global i32 add (i32 sub (i32 ptrtoint (i32** @localgotequiv to i32),
                             i32 ptrtoint (i32* @deltaplus to i32)),
                             i32 55)

define i32 @t0(i32 %a) {
  %x = add i32 sub (i32 ptrtoint (i32** @bargotequiv to i32),
                    i32 ptrtoint (i32 (i32)* @t0 to i32)), %a
  ret i32 %x
}

; Test indirect local symbols.
; CHECK-LABEL: _localindirect
; CHECK: .long 65603
@localindirect = internal constant i32  65603
@got.localindirect = private unnamed_addr constant i32* @localindirect

; CHECK-LABEL: _localindirectuser:
; CHECK: .long   L_localindirect$non_lazy_ptr-_localindirectuser
@localindirectuser = internal constant
  i32 sub (i32 ptrtoint (i32** @got.localindirect to i32),
           i32 ptrtoint (i32* @localindirectuser to i32))

; Test internal indirect local symbols where the user doesn't see the
; definition of the other symbols yet.

; We used to check if the symbol is defined and not external to guess if it has
; local linkage, but that doesn't work if the symbol is defined after. The code
; should check if the GlobalValue itself has local linkage.

; CHECK-LABEL: _undeflocalindirectuser:
; CHECK: .long L_undeflocalindirect$non_lazy_ptr-_undeflocalindirectuser
@undeflocalindirectuser = internal constant
  i32 sub (i32 ptrtoint (i32** @got.undeflocalindirect to i32),
           i32 ptrtoint (i32* @undeflocalindirectuser to i32)),
  section "__TEXT,__const"

; CHECK-LABEL: _undeflocalindirect:
; CHECK: .long 65603
@undeflocalindirect = internal constant i32  65603
@got.undeflocalindirect = private unnamed_addr constant i32* @undeflocalindirect

; CHECK-LABEL: .section __IMPORT,__pointers

; CHECK-LABEL: L_localfoo$non_lazy_ptr:
; CHECK: .indirect_symbol _localfoo
; CHECK-NOT: .long _localfoo
; CHECK-NEXT: .long 0

; CHECK-LABEL: L_localindirect$non_lazy_ptr:
; CHECK: .indirect_symbol _localindirect
; CHECK-NOT: .long 0
; CHECK-NEXT: .long _localindirect

; CHECK-LABEL: L_undeflocalindirect$non_lazy_ptr:
; CHECK: .indirect_symbol _undeflocalindirect
; CHECK-NOT: .long 0
; CHECK-NEXT: .long _undeflocalindirect
