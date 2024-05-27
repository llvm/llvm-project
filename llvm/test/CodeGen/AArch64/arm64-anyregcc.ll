; RUN: llc < %s -debug-entry-values -mtriple=arm64-apple-darwin | FileCheck %s

; Stackmap Header: no constants - 18 callsites
; CHECK-LABEL: .section	__LLVM_STACKMAPS,__llvm_stackmaps
; CHECK-NEXT:  __LLVM_StackMaps:
; Header
; CHECK-NEXT:   .byte 3
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 0
; Num Functions
; CHECK-NEXT:   .long 18
; Num LargeConstants
; CHECK-NEXT:   .long 0
; Num Callsites
; CHECK-NEXT:   .long 18

; Functions and stack size
; CHECK-NEXT:   .quad _test
; CHECK-NEXT:   .quad 16
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad _property_access1
; CHECK-NEXT:   .quad 16
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad _property_access2
; CHECK-NEXT:   .quad 32
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad _property_access3
; CHECK-NEXT:   .quad 32
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad _anyreg_test1
; CHECK-NEXT:   .quad 16
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad _anyreg_test2
; CHECK-NEXT:   .quad 16
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad _patchpoint_spilldef
; CHECK-NEXT:   .quad 112
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad _patchpoint_spillargs
; CHECK-NEXT:   .quad 128
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad _generic_test_i32
; CHECK-NEXT:   .quad 16
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad _generic_test_i64
; CHECK-NEXT:   .quad 16
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad _generic_test_p0
; CHECK-NEXT:   .quad 16
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad _generic_test_f16
; CHECK-NEXT:   .quad 16
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad _generic_test_f32
; CHECK-NEXT:   .quad 16
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad _generic_test_f64
; CHECK-NEXT:   .quad 16
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad _generic_test_v16i8
; CHECK-NEXT:   .quad 16
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad _generic_test_v4i32
; CHECK-NEXT:   .quad 16
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad _generic_test_v4f32
; CHECK-NEXT:   .quad 16
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad _generic_test_v2f64
; CHECK-NEXT:   .quad 16
; CHECK-NEXT:   .quad 1


; test
; CHECK-LABEL:  .long   L{{.*}}-_test
; CHECK-NEXT:   .short  0
; 3 locations
; CHECK-NEXT:   .short  3
; Loc 0: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 4
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 1: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 4
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 2: Constant 3
; CHECK-NEXT:   .byte 4
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 3
define i64 @test() nounwind ssp uwtable {
entry:
  call anyregcc void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 0, i32 16, ptr null, i32 2, i32 1, i32 2, i64 3)
  ret i64 0
}

; property access 1 - %obj is an anyreg call argument and should therefore be in a register
; CHECK-LABEL:  .long   L{{.*}}-_property_access1
; CHECK-NEXT:   .short 0
; 2 locations
; CHECK-NEXT:   .short  2
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 1: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
define i64 @property_access1(ptr %obj) nounwind ssp uwtable {
entry:
  %f = inttoptr i64 281474417671919 to ptr
  %ret = call anyregcc i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 1, i32 20, ptr %f, i32 1, ptr %obj)
  ret i64 %ret
}

; property access 2 - %obj is an anyreg call argument and should therefore be in a register
; CHECK-LABEL:  .long   L{{.*}}-_property_access2
; CHECK-NEXT:   .short 0
; 2 locations
; CHECK-NEXT:   .short  2
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 1: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
define i64 @property_access2() nounwind ssp uwtable {
entry:
  %obj = alloca i64, align 8
  %f = inttoptr i64 281474417671919 to ptr
  %ret = call anyregcc i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 2, i32 20, ptr %f, i32 1, ptr %obj)
  ret i64 %ret
}

; property access 3 - %obj is a frame index
; CHECK-LABEL:  .long   L{{.*}}-_property_access3
; CHECK-NEXT:   .short 0
; 2 locations
; CHECK-NEXT:   .short  2
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 1: Direct FP - 8
; CHECK-NEXT:   .byte 2
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short 29
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long -8
define i64 @property_access3() nounwind ssp uwtable {
entry:
  %obj = alloca i64, align 8
  %f = inttoptr i64 281474417671919 to ptr
  %ret = call anyregcc i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 3, i32 20, ptr %f, i32 0, ptr %obj)
  ret i64 %ret
}

; anyreg_test1
; CHECK-LABEL:  .long   L{{.*}}-_anyreg_test1
; CHECK-NEXT:   .short 0
; 14 locations
; CHECK-NEXT:   .short  14
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 1: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 2: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 3: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 4: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 5: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 6: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 7: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 8: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 9: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 10: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 11: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 12: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 13: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
define i64 @anyreg_test1(ptr %a1, ptr %a2, ptr %a3, ptr %a4, ptr %a5, ptr %a6, ptr %a7, ptr %a8, ptr %a9, ptr %a10, ptr %a11, ptr %a12, ptr %a13) nounwind ssp uwtable {
entry:
  %f = inttoptr i64 281474417671919 to ptr
  %ret = call anyregcc i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 4, i32 20, ptr %f, i32 13, ptr %a1, ptr %a2, ptr %a3, ptr %a4, ptr %a5, ptr %a6, ptr %a7, ptr %a8, ptr %a9, ptr %a10, ptr %a11, ptr %a12, ptr %a13)
  ret i64 %ret
}

; anyreg_test2
; CHECK-LABEL:  .long   L{{.*}}-_anyreg_test2
; CHECK-NEXT:   .short 0
; 14 locations
; CHECK-NEXT:   .short  14
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 1: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 2: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 3: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 4: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 5: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 6: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 7: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 8: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 9: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 10: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 11: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 12: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 13: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
define i64 @anyreg_test2(ptr %a1, ptr %a2, ptr %a3, ptr %a4, ptr %a5, ptr %a6, ptr %a7, ptr %a8, ptr %a9, ptr %a10, ptr %a11, ptr %a12, ptr %a13) nounwind ssp uwtable {
entry:
  %f = inttoptr i64 281474417671919 to ptr
  %ret = call anyregcc i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 5, i32 20, ptr %f, i32 8, ptr %a1, ptr %a2, ptr %a3, ptr %a4, ptr %a5, ptr %a6, ptr %a7, ptr %a8, ptr %a9, ptr %a10, ptr %a11, ptr %a12, ptr %a13)
  ret i64 %ret
}

; Test spilling the return value of an anyregcc call.
;
; <rdar://problem/15432754> [JS] Assertion: "Folded a def to a non-store!"
;
; CHECK-LABEL: .long L{{.*}}-_patchpoint_spilldef
; CHECK-NEXT: .short 0
; CHECK-NEXT: .short 3
; Loc 0: Register (some register that will be spilled to the stack)
; CHECK-NEXT: .byte  1
; CHECK-NEXT: .byte 0
; CHECK-NEXT: .short 8
; CHECK-NEXT: .short {{[0-9]+}}
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long  0
; Loc 1: Register
; CHECK-NEXT: .byte  1
; CHECK-NEXT: .byte 0
; CHECK-NEXT: .short 8
; CHECK-NEXT: .short {{[0-9]+}}
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long  0
; Loc 1: Register
; CHECK-NEXT: .byte  1
; CHECK-NEXT: .byte 0
; CHECK-NEXT: .short 8
; CHECK-NEXT: .short {{[0-9]+}}
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long  0
define i64 @patchpoint_spilldef(i64 %p1, i64 %p2, i64 %p3, i64 %p4) {
entry:
  %result = tail call anyregcc i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 12, i32 16, ptr inttoptr (i64 0 to ptr), i32 2, i64 %p1, i64 %p2)
  tail call void asm sideeffect "nop", "~{x0},~{x1},~{x2},~{x3},~{x4},~{x5},~{x6},~{x7},~{x8},~{x9},~{x10},~{x11},~{x12},~{x13},~{x14},~{x15},~{x16},~{x17},~{x18},~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27},~{x28},~{x29},~{x30},~{x31}"() nounwind
  ret i64 %result
}

; Test spilling the arguments of an anyregcc call.
;
; <rdar://problem/15487687> [JS] AnyRegCC argument ends up being spilled
;
; CHECK-LABEL: .long L{{.*}}-_patchpoint_spillargs
; CHECK-NEXT: .short 0
; CHECK-NEXT: .short 5
; Loc 0: Return a register
; CHECK-NEXT: .byte  1
; CHECK-NEXT: .byte 0
; CHECK-NEXT: .short 8
; CHECK-NEXT: .short {{[0-9]+}}
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long  0
; Loc 1: Arg0 in a Register
; CHECK-NEXT: .byte  1
; CHECK-NEXT: .byte 0
; CHECK-NEXT: .short 8
; CHECK-NEXT: .short {{[0-9]+}}
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long  0
; Loc 2: Arg1 in a Register
; CHECK-NEXT: .byte  1
; CHECK-NEXT: .byte 0
; CHECK-NEXT: .short 8
; CHECK-NEXT: .short {{[0-9]+}}
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long  0
; Loc 3: Arg2 spilled to FP -96
; CHECK-NEXT: .byte  3
; CHECK-NEXT: .byte 0
; CHECK-NEXT: .short 8
; CHECK-NEXT: .short 29
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long -96
; Loc 4: Arg3 spilled to FP - 88
; CHECK-NEXT: .byte  3
; CHECK-NEXT: .byte 0
; CHECK-NEXT: .short 8
; CHECK-NEXT: .short 29
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long -88
define i64 @patchpoint_spillargs(i64 %p1, i64 %p2, i64 %p3, i64 %p4) {
entry:
  tail call void asm sideeffect "nop", "~{x0},~{x1},~{x2},~{x3},~{x4},~{x5},~{x6},~{x7},~{x8},~{x9},~{x10},~{x11},~{x12},~{x13},~{x14},~{x15},~{x16},~{x17},~{x18},~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27},~{x28},~{x29},~{x30},~{x31}"() nounwind
  %result = tail call anyregcc i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 13, i32 16, ptr inttoptr (i64 0 to ptr), i32 2, i64 %p1, i64 %p2, i64 %p3, i64 %p4)
  ret i64 %result
}

; generic_test_i32
; CHECK-LABEL:  .long   L{{.*}}-_generic_test_i32
; CHECK-NEXT:   .short  0
; 1 location
; CHECK-NEXT:   .short  1
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 4
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
define i32 @generic_test_i32() nounwind ssp uwtable {
entry:
  %ret = call anyregcc i32 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i32(i64 14, i32 20, ptr null, i32 0)
  ret i32 %ret
}

; generic_test_i64
; CHECK-LABEL:  .long   L{{.*}}-_generic_test_i64
; CHECK-NEXT:   .short  0
; 1 location
; CHECK-NEXT:   .short  1
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
define i64 @generic_test_i64() nounwind ssp uwtable {
entry:
  %ret = call anyregcc i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 14, i32 20, ptr null, i32 0)
  ret i64 %ret
}

; generic_test_p0
; CHECK-LABEL:  .long   L{{.*}}-_generic_test_p0
; CHECK-NEXT:   .short  0
; 1 location
; CHECK-NEXT:   .short  1
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
define ptr @generic_test_p0() nounwind ssp uwtable {
entry:
  %ret = call anyregcc ptr (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.p0(i64 14, i32 20, ptr null, i32 0)
  ret ptr %ret
}

; generic_test_f16
; CHECK-LABEL:  .long   L{{.*}}-_generic_test_f16
; CHECK-NEXT:   .short  0
; 1 location
; CHECK-NEXT:   .short  1
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 2
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
define half @generic_test_f16() nounwind ssp uwtable {
entry:
  %ret = call anyregcc half (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.f16(i64 14, i32 20, ptr null, i32 0)
  ret half %ret
}

; generic_test_f32
; CHECK-LABEL:  .long   L{{.*}}-_generic_test_f32
; CHECK-NEXT:   .short  0
; 1 location
; CHECK-NEXT:   .short  1
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 4
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
define float @generic_test_f32() nounwind ssp uwtable {
entry:
  %ret = call anyregcc float (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.f32(i64 14, i32 20, ptr null, i32 0)
  ret float %ret
}

; generic_test_f64
; CHECK-LABEL:  .long   L{{.*}}-_generic_test_f64
; CHECK-NEXT:   .short  0
; 1 location
; CHECK-NEXT:   .short  1
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
define double @generic_test_f64() nounwind ssp uwtable {
entry:
  %ret = call anyregcc double (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.f64(i64 14, i32 20, ptr null, i32 0)
  ret double %ret
}

; generic_test_v16i8
; CHECK-LABEL:  .long   L{{.*}}-_generic_test_v16i8
; CHECK-NEXT:   .short  0
; 1 location
; CHECK-NEXT:   .short  1
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 16
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
define <16 x i8> @generic_test_v16i8() nounwind ssp uwtable {
entry:
  %ret = call anyregcc <16 x i8> (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.v16i8(i64 14, i32 20, ptr null, i32 0)
  ret <16 x i8> %ret
}

; generic_test_v4i32
; CHECK-LABEL:  .long   L{{.*}}-_generic_test_v4i32
; CHECK-NEXT:   .short  0
; 1 location
; CHECK-NEXT:   .short  1
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 16
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
define <4 x i32> @generic_test_v4i32() nounwind ssp uwtable {
entry:
  %ret = call anyregcc <4 x i32> (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.v4i32(i64 14, i32 20, ptr null, i32 0)
  ret <4 x i32> %ret
}

; generic_test_v4f32
; CHECK-LABEL:  .long   L{{.*}}-_generic_test_v4f32
; CHECK-NEXT:   .short  0
; 1 location
; CHECK-NEXT:   .short  1
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 16
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
define <4 x float> @generic_test_v4f32() nounwind ssp uwtable {
entry:
  %ret = call anyregcc <4 x float> (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.v4f32(i64 14, i32 20, ptr null, i32 0)
  ret <4 x float> %ret
}

; generic_test_v2f64
; CHECK-LABEL:  .long   L{{.*}}-_generic_test_v2f64
; CHECK-NEXT:   .short  0
; 1 location
; CHECK-NEXT:   .short  1
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 16
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
define <2 x double> @generic_test_v2f64() nounwind ssp uwtable {
entry:
  %ret = call anyregcc <2 x double> (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.v2f64(i64 14, i32 20, ptr null, i32 0)
  ret <2 x double> %ret
}

declare void @llvm.experimental.patchpoint.void(i64, i32, ptr, i32, ...)
declare i32 @llvm.experimental.patchpoint.i32(i64, i32, ptr, i32, ...)
declare i64 @llvm.experimental.patchpoint.i64(i64, i32, ptr, i32, ...)
declare ptr @llvm.experimental.patchpoint.p0(i64, i32, ptr, i32, ...)
declare half @llvm.experimental.patchpoint.f16(i64, i32, ptr, i32, ...)
declare float @llvm.experimental.patchpoint.f32(i64, i32, ptr, i32, ...)
declare double @llvm.experimental.patchpoint.f64(i64, i32, ptr, i32, ...)
declare <16 x i8> @llvm.experimental.patchpoint.v16i8(i64, i32, ptr, i32, ...)
declare <4 x i32> @llvm.experimental.patchpoint.v4i32(i64, i32, ptr, i32, ...)
declare <4 x float> @llvm.experimental.patchpoint.v4f32(i64, i32, ptr, i32, ...)
declare <2 x double> @llvm.experimental.patchpoint.v2f64(i64, i32, ptr, i32, ...)
