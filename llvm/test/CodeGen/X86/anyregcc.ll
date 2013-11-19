; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s

; Stackmap Header: no constants - 6 callsites
; CHECK-LABEL: .section	__LLVM_STACKMAPS,__llvm_stackmaps
; CHECK-NEXT:  __LLVM_StackMaps:
; Header
; CHECK-NEXT:   .long   0
; Num Constants
; CHECK-NEXT:   .long   0
; Num Callsites
; CHECK-NEXT:   .long   8

; test
; CHECK-NEXT:   .long   0
; CHECK-LABEL:  .long   L{{.*}}-_test
; CHECK-NEXT:   .short  0
; 3 locations
; CHECK-NEXT:   .short  3
; Loc 0: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 4
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 1: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 4
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 2: Constant 3
; CHECK-NEXT:   .byte 4
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .long 3
define i64 @test() nounwind ssp uwtable {
entry:
  call anyregcc void (i32, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.void(i32 0, i32 15, i8* null, i32 2, i32 1, i32 2, i64 3)
  ret i64 0
}

; property access 1 - %obj is an anyreg call argument and should therefore be in a register
; CHECK-NEXT:   .long   1
; CHECK-LABEL:  .long   L{{.*}}-_property_access1
; CHECK-NEXT:   .short  0
; 2 locations
; CHECK-NEXT:   .short  2
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 1: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
define i64 @property_access1(i8* %obj) nounwind ssp uwtable {
entry:
  %f = inttoptr i64 12297829382473034410 to i8*
  %ret = call anyregcc i64 (i32, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.i64(i32 1, i32 15, i8* %f, i32 1, i8* %obj)
  ret i64 %ret
}

; property access 2 - %obj is an anyreg call argument and should therefore be in a register
; CHECK-NEXT:   .long   2
; CHECK-LABEL:  .long   L{{.*}}-_property_access2
; CHECK-NEXT:   .short  0
; 2 locations
; CHECK-NEXT:   .short  2
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 1: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
define i64 @property_access2() nounwind ssp uwtable {
entry:
  %obj = alloca i64, align 8
  %f = inttoptr i64 12297829382473034410 to i8*
  %ret = call anyregcc i64 (i32, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.i64(i32 2, i32 15, i8* %f, i32 1, i64* %obj)
  ret i64 %ret
}

; property access 3 - %obj is a frame index
; CHECK-NEXT:   .long   3
; CHECK-LABEL:  .long   L{{.*}}-_property_access3
; CHECK-NEXT:   .short  0
; 2 locations
; CHECK-NEXT:   .short  2
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 1: Register <-- this will be folded once folding for FI is implemented
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
define i64 @property_access3() nounwind ssp uwtable {
entry:
  %obj = alloca i64, align 8
  %f = inttoptr i64 12297829382473034410 to i8*
  %ret = call anyregcc i64 (i32, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.i64(i32 3, i32 15, i8* %f, i32 0, i64* %obj)
  ret i64 %ret
}

; anyreg_test1
; CHECK-NEXT:   .long   4
; CHECK-LABEL:  .long   L{{.*}}-_anyreg_test1
; CHECK-NEXT:   .short  0
; 14 locations
; CHECK-NEXT:   .short  14
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 1: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 2: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 3: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 4: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 5: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 6: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 7: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 8: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 9: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 10: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 11: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 12: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 13: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
define i64 @anyreg_test1(i8* %a1, i8* %a2, i8* %a3, i8* %a4, i8* %a5, i8* %a6, i8* %a7, i8* %a8, i8* %a9, i8* %a10, i8* %a11, i8* %a12, i8* %a13) nounwind ssp uwtable {
entry:
  %f = inttoptr i64 12297829382473034410 to i8*
  %ret = call anyregcc i64 (i32, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.i64(i32 4, i32 15, i8* %f, i32 13, i8* %a1, i8* %a2, i8* %a3, i8* %a4, i8* %a5, i8* %a6, i8* %a7, i8* %a8, i8* %a9, i8* %a10, i8* %a11, i8* %a12, i8* %a13)
  ret i64 %ret
}

; anyreg_test2
; CHECK-NEXT:   .long   5
; CHECK-LABEL:  .long   L{{.*}}-_anyreg_test2
; CHECK-NEXT:   .short  0
; 14 locations
; CHECK-NEXT:   .short  14
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 1: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 2: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 3: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 4: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 5: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 6: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 7: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 8: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 9: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 10: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 11: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 12: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 13: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
define i64 @anyreg_test2(i8* %a1, i8* %a2, i8* %a3, i8* %a4, i8* %a5, i8* %a6, i8* %a7, i8* %a8, i8* %a9, i8* %a10, i8* %a11, i8* %a12, i8* %a13) nounwind ssp uwtable {
entry:
  %f = inttoptr i64 12297829382473034410 to i8*
  %ret = call anyregcc i64 (i32, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.i64(i32 5, i32 15, i8* %f, i32 8, i8* %a1, i8* %a2, i8* %a3, i8* %a4, i8* %a5, i8* %a6, i8* %a7, i8* %a8, i8* %a9, i8* %a10, i8* %a11, i8* %a12, i8* %a13)
  ret i64 %ret
}

; Test spilling the return value of an anyregcc call.
;
; <rdar://problem/15432754> [JS] Assertion: "Folded a def to a non-store!"
;
; CHECK-LABEL: .long 12
; CHECK-LABEL: .long L{{.*}}-_patchpoint_spilldef
; CHECK-NEXT: .short 0
; CHECK-NEXT: .short 3
; Loc 0: Register (some register that will be spilled to the stack)
; CHECK-NEXT: .byte  1
; CHECK-NEXT: .byte  8
; CHECK-NEXT: .short {{[0-9]+}}
; CHECK-NEXT: .long  0
; Loc 1: Register RDI
; CHECK-NEXT: .byte  1
; CHECK-NEXT: .byte  8
; CHECK-NEXT: .short 5
; CHECK-NEXT: .long  0
; Loc 1: Register RSI
; CHECK-NEXT: .byte  1
; CHECK-NEXT: .byte  8
; CHECK-NEXT: .short 4
; CHECK-NEXT: .long  0
define i64 @patchpoint_spilldef(i64 %p1, i64 %p2, i64 %p3, i64 %p4) {
entry:
  %result = tail call anyregcc i64 (i32, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.i64(i32 12, i32 15, i8* inttoptr (i64 0 to i8*), i32 2, i64 %p1, i64 %p2)
  tail call void asm sideeffect "nop", "~{ax},~{bx},~{cx},~{dx},~{bp},~{si},~{di},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"() nounwind
  ret i64 %result
}

; Test spilling the arguments of an anyregcc call.
;
; <rdar://problem/15487687> [JS] AnyRegCC argument ends up being spilled
;
; CHECK-LABEL: .long 13
; CHECK-LABEL: .long L{{.*}}-_patchpoint_spillargs
; CHECK-NEXT: .short 0
; CHECK-NEXT: .short 5
; Loc 0: Return a register
; CHECK-NEXT: .byte  1
; CHECK-NEXT: .byte  8
; CHECK-NEXT: .short {{[0-9]+}}
; CHECK-NEXT: .long  0
; Loc 1: Arg0 in a Register
; CHECK-NEXT: .byte  1
; CHECK-NEXT: .byte  8
; CHECK-NEXT: .short {{[0-9]+}}
; CHECK-NEXT: .long  0
; Loc 2: Arg1 in a Register
; CHECK-NEXT: .byte  1
; CHECK-NEXT: .byte  8
; CHECK-NEXT: .short {{[0-9]+}}
; CHECK-NEXT: .long  0
; Loc 3: Arg2 spilled to RBP +
; CHECK-NEXT: .byte  3
; CHECK-NEXT: .byte  8
; CHECK-NEXT: .short 7
; CHECK-NEXT: .long  {{[0-9]+}}
; Loc 4: Arg3 spilled to RBP +
; CHECK-NEXT: .byte  3
; CHECK-NEXT: .byte  8
; CHECK-NEXT: .short 7
; CHECK-NEXT: .long  {{[0-9]+}}
define i64 @patchpoint_spillargs(i64 %p1, i64 %p2, i64 %p3, i64 %p4) {
entry:
  tail call void asm sideeffect "nop", "~{ax},~{bx},~{cx},~{dx},~{bp},~{si},~{di},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"() nounwind
  %result = tail call anyregcc i64 (i32, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.i64(i32 13, i32 15, i8* inttoptr (i64 0 to i8*), i32 2, i64 %p1, i64 %p2, i64 %p3, i64 %p4)
  ret i64 %result
}

declare void @llvm.experimental.patchpoint.void(i32, i32, i8*, i32, ...)
declare i64 @llvm.experimental.patchpoint.i64(i32, i32, i8*, i32, ...)
