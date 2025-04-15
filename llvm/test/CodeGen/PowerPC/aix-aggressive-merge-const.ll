; RUN: llc --verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff \
; RUN: -global-merge-all-const=true < %s | FileCheck %s

; RUN: llc --verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff \
; RUN: -global-merge-all-const=false < %s | FileCheck --check-prefix=NOMERGE %s

%struct.pc_t = type { i8 }
%struct.S = type { i32, i32, i32, i32, [9 x i32] }

@constinit = private unnamed_addr constant <{ i32, i32, i32, i32, [9 x i32] }> <{ i32 0, i32 0, i32 0, i32 2, [9 x i32] zeroinitializer }>, align 4
@.str = private unnamed_addr constant [6 x i8] c"hello\00", align 1
@.str.1 = private unnamed_addr constant [6 x i8] c"world\00", align 1
@.str.2 = private unnamed_addr constant [6 x i8] c"abcde\00", align 1
@.str.3 = private unnamed_addr constant [6 x i8] c"fghij\00", align 1
@pc = internal constant %struct.pc_t zeroinitializer, align 1
@s = internal constant %struct.S { i32 1, i32 2, i32 3, i32 4, [9 x i32] [i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13] }, align 4

; Function Attrs: mustprogress
define noundef i32 @f5() {
entry:
  %call = tail call noundef i32 @f4(ptr noundef nonnull @pc)
  ret i32 %call
}

declare noundef i32 @f4(ptr noundef)
declare noundef i32 @printf(ptr nocapture noundef readonly, ...)

define noundef i32 @f1() {
entry:
  %call = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str)
  ret i32 %call
}


; Function Attrs: mustprogress nofree nounwind
define noundef i32 @f2() {
entry:
  %call = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1)
  ret i32 %call
}

define noundef i32 @f3() {
entry:
  %call = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2)
  ret i32 %call
}

define noundef i32 @f7() {
entry:
  %call = tail call noundef i32 @f6(ptr noundef nonnull @s)
  ret i32 %call
}

declare noundef i32 @f6(ptr noundef)

; CHECK:             .csect L.._MergedGlobals[RO],2
; CHECK-NEXT:        .lglobl pc                          # @_MergedGlobals
; CHECK-NEXT:        .lglobl s
; CHECK-NEXT:        .align  2
; CHECK-NEXT:pc:
; CHECK-NEXT:        .space  1
; CHECK-NEXT:L...str:
; CHECK-NEXT:        .string "hello"
; CHECK-NEXT:L...str.1:
; CHECK-NEXT:        .string "world"
; CHECK-NEXT:L...str.2:
; CHECK-NEXT:        .string "abcde"
; CHECK-NEXT:L...str.3:
; CHECK-NEXT:        .string "fghij"
; CHECK-NEXT:        .space  3
; CHECK-NEXT:L..constinit:
; CHECK-NEXT:        .vbyte  4, 0                            # 0x0
; CHECK-NEXT:        .vbyte  4, 0                            # 0x0
; CHECK-NEXT:        .vbyte  4, 0                            # 0x0
; CHECK-NEXT:        .vbyte  4, 2                            # 0x2
; CHECK-NEXT:        .space  36
; CHECK-NEXT:s:
; CHECK-NEXT:        .vbyte  4, 1                            # 0x1
; CHECK-NEXT:        .vbyte  4, 2                            # 0x2
; CHECK-NEXT:        .vbyte  4, 3                            # 0x3
; CHECK-NEXT:        .vbyte  4, 4                            # 0x4
; CHECK-NEXT:        .vbyte  4, 5                            # 0x5
; CHECK-NEXT:        .vbyte  4, 6                            # 0x6
; CHECK-NEXT:        .vbyte  4, 7                            # 0x7
; CHECK-NEXT:        .vbyte  4, 8                            # 0x8
; CHECK-NEXT:        .vbyte  4, 9                            # 0x9
; CHECK-NEXT:        .vbyte  4, 10                           # 0xa
; CHECK-NEXT:        .vbyte  4, 11                           # 0xb
; CHECK-NEXT:        .vbyte  4, 12                           # 0xc
; CHECK-NEXT:        .vbyte  4, 13                           # 0xd


; NOMERGE-NOT: L.._MergedGGlobals[RO]
