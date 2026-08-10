; RUN: llc -mtriple=arm-apple-darwin %s -o %t
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
@localgotequiv = private unnamed_addr constant ptr @localfoo

@extfoo = external global i32
@extgotequiv = private unnamed_addr constant ptr @extfoo

; Don't replace GOT equivalent usage within instructions and emit the GOT
; equivalent since it can't be replaced by the GOT entry. @bargotequiv is
; used by an instruction inside @t0.
;
; CHECK: l_bargotequiv:
; CHECK-NEXT:  .long   _extbar
@extbar = external global i32
@bargotequiv = private unnamed_addr constant ptr @extbar

@table = global [4 x %struct.data] [
; CHECK-LABEL: _table
  %struct.data { i32 1, %struct.anon { i32 2, i32 3 } },
; Test GOT equivalent usage inside nested constant arrays.
; CHECK: .long   5
; CHECK-NOT: l_localgotequiv-(_table+20)
; CHECK-NEXT: L_localfoo$non_lazy_ptr-(_table+20)
  %struct.data { i32 4, %struct.anon { i32 5,
    i32 sub (i32 ptrtoint (ptr @localgotequiv to i32),
             i32 ptrtoint (ptr getelementptr inbounds ([4 x %struct.data], ptr @table, i32 0, i32 1, i32 1, i32 1) to i32))}
  },
; CHECK: .long   5
; CHECK-NOT: l_extgotequiv-(_table+32)
; CHECK-NEXT: L_extfoo$non_lazy_ptr-(_table+32)
  %struct.data { i32 4, %struct.anon { i32 5,
    i32 sub (i32 ptrtoint (ptr @extgotequiv to i32),
             i32 ptrtoint (ptr getelementptr inbounds ([4 x %struct.data], ptr @table, i32 0, i32 2, i32 1, i32 1) to i32))}
  },
; Test support for arbitrary constants into the GOTPCREL offset
; CHECK: .long   5
; CHECK-NOT: (l_extgotequiv-(_table+44))+24
; CHECK-NEXT: L_extfoo$non_lazy_ptr-(_table+20)
  %struct.data { i32 4, %struct.anon { i32 5,
    i32 add (i32 sub (i32 ptrtoint (ptr @extgotequiv to i32),
                      i32 ptrtoint (ptr getelementptr inbounds ([4 x %struct.data], ptr @table, i32 0, i32 3, i32 1, i32 1) to i32)),
                      i32 24)}
  }
], align 16

; Test multiple uses of GOT equivalents.
; CHECK-LABEL: _delta
; CHECK: .long   L_extfoo$non_lazy_ptr-_delta
@delta = global i32 sub (i32 ptrtoint (ptr @extgotequiv to i32),
                         i32 ptrtoint (ptr @delta to i32))

; CHECK-LABEL: _deltaplus:
; CHECK: .long   L_localfoo$non_lazy_ptr-(_deltaplus-55)
@deltaplus = global i32 add (i32 sub (i32 ptrtoint (ptr @localgotequiv to i32),
                             i32 ptrtoint (ptr @deltaplus to i32)),
                             i32 55)

define i32 @t0(i32 %a) {
  %x = add i32 sub (i32 ptrtoint (ptr @bargotequiv to i32),
                    i32 ptrtoint (ptr @t0 to i32)), %a
  ret i32 %x
}
