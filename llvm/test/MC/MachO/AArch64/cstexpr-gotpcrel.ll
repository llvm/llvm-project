; RUN: llc -mtriple=arm64-apple-darwin %s -o - | FileCheck %s

; GOT equivalent globals references can be replaced by the GOT entry of the
; final symbol instead.

%struct.data = type { i32, %struct.anon }
%struct.anon = type { i32, i32 }

@localfoo = global i32 42
@localgotequiv = private unnamed_addr constant ptr @localfoo

@extfoo = external global i32
@extgotequiv = private unnamed_addr constant ptr @extfoo

; Don't replace GOT equivalent usage within instructions and emit the GOT
; equivalent since it can't be replaced by the GOT entry. @bargotequiv is
; used by an instruction inside @t0.
;
; CHECK: l_bargotequiv:
; CHECK-NEXT:  .quad   _extbar
@extbar = external global i32
@bargotequiv = private unnamed_addr constant ptr @extbar

@table = global [4 x %struct.data] [
; CHECK-LABEL: _table
  %struct.data { i32 1, %struct.anon { i32 2, i32 3 } },
; Test GOT equivalent usage inside nested constant arrays.

; CHECK: .long   5
; CHECK-NOT: .long   _localgotequiv-(_table+20)
; CHECK-NEXT: Ltmp0:
; CHECK-NEXT: .long _localfoo@GOT-Ltmp0
  %struct.data { i32 4, %struct.anon { i32 5,
    i32 trunc (i64 sub (i64 ptrtoint (ptr @localgotequiv to i64),
                        i64 ptrtoint (ptr getelementptr inbounds ([4 x %struct.data], ptr @table, i32 0, i64 1, i32 1, i32 1) to i64))
                        to i32)}
  },

; CHECK: .long   5
; CHECK-NOT: _extgotequiv-(_table+32)
; CHECK-NEXT: Ltmp1:
; CHECK-NEXT: _extfoo@GOT-Ltmp1
  %struct.data { i32 4, %struct.anon { i32 5,
    i32 trunc (i64 sub (i64 ptrtoint (ptr @extgotequiv to i64),
                        i64 ptrtoint (ptr getelementptr inbounds ([4 x %struct.data], ptr @table, i32 0, i64 2, i32 1, i32 1) to i64))
                        to i32)}
  },
; Test support for arbitrary constants into the GOTPCREL offset, which is
; supported on x86-64 but not on ARM64

; CHECK: .long   5
; CHECK-NEXT: .long ((l_extgotequiv-_table)-44)+24
  %struct.data { i32 4, %struct.anon { i32 5,
    i32 add (i32 trunc (i64 sub (i64 ptrtoint (ptr @extgotequiv to i64),
                                 i64 ptrtoint (ptr getelementptr inbounds ([4 x %struct.data], ptr @table, i32 0, i64 3, i32 1, i32 1) to i64))
                                 to i32), i32 24)}
  }
], align 16

; Test multiple uses of GOT equivalents.

; CHECK-LABEL: _delta
; CHECK: Ltmp2:
; CHECK-NEXT:  .long _extfoo@GOT-Ltmp2
@delta = global i32 trunc (i64 sub (i64 ptrtoint (ptr @extgotequiv to i64),
                                    i64 ptrtoint (ptr @delta to i64))
                           to i32)

; CHECK-LABEL: _deltaplus:
; CHECK: .long  (l_localgotequiv-_deltaplus)+55
@deltaplus = global i32 add (i32 trunc (i64 sub (i64 ptrtoint (ptr @localgotequiv to i64),
                                        i64 ptrtoint (ptr @deltaplus to i64))
                                        to i32), i32 55)

define i32 @t0(i32 %a) {
  %x = add i32 trunc (i64 sub (i64 ptrtoint (ptr @bargotequiv to i64),
                               i64 ptrtoint (ptr @t0 to i64))
                           to i32), %a
  ret i32 %x
}

; Check that these got equivalent symbols are emitted on ARM64. Since ARM64
; does not support encoding an extra offset with @GOT, we still need to emit the
; equivalents for use by such IR constructs. Check them at the end of the test
; since they will start out as GOT equivalent candidates, but they are actually
; needed and are therefore emitted at the end.

; CHECK-LABEL: l_localgotequiv:
; CHECK-NEXT: .quad   _localfoo

; CHECK-LABEL: l_extgotequiv:
; CHECK-NEXT: .quad   _extfoo
