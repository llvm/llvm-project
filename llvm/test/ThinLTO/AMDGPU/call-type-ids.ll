;; Test the call type IDs in the module summary of an AMDGPU module, in the
;; per-module summary and in the combined index. AMDGPU call type IDs compare
;; the sizes of the types, not the IR types. The summary stores the MD5 hash of
;; each ID:
;;   798791889831736034   = "amdgpu.v1:cc0:l"
;;   3430040695729964317  = "amdgpu.v1:cc0:ii"
;;   5485679292981660066  = "amdgpu.v1:cc0:v"
;;   5970146889962867861  = "amdgpu.v1:cc0:vl"
;;   11963068178503280946 = "amdgpu.v1:cc100:vl"

; RUN: opt -mtriple=amdgcn-amd-amdhsa -module-summary %s -o %t.bc
; RUN: llvm-dis %t.bc -o - | FileCheck %s
; RUN: llvm-bcanalyzer -dump %t.bc | FileCheck %s --check-prefix=BCAN
; RUN: llvm-lto2 run %t.bc -o %t.o -thinlto-distributed-indexes \
; RUN:   -r=%t.bc,f_float,px -r=%t.bc,f_i32,px -r=%t.bc,device,px \
; RUN:   -r=%t.bc,gfx,px -r=%t.bc,kernel,px -r=%t.bc,fp,px -r=%t.bc,ifn,px \
; RUN:   -r=%t.bc,ifn_alias,px -r=%t.bc,resolver,px -r=%t.bc,ifunc_caller,px \
; RUN:   -r=%t.bc,asm_goto,px -r=%t.bc,asm_invoke,px -r=%t.bc,personality, \
; RUN:   -r=%t.bc,const_callers,px
; RUN: llvm-dis %t.bc.thinlto.bc -o - | FileCheck %s --check-prefix=COMBINED

;; A target that does not use call type IDs gets none.
; RUN: opt -module-summary %s -o %t.none.bc
; RUN: llvm-dis %t.none.bc -o - | FileCheck %s --check-prefix=NONE --implicit-check-not=callTypeIdInfo

;; float(float) and i32(i32) have the same call type ID, and the calling
;; convention is part of the ID.
; CHECK-DAG: (name: "f_float", {{.*}}, callTypeIdInfo: (callTypeId: 3430040695729964317))))
; CHECK-DAG: (name: "f_i32", {{.*}}, callTypeIdInfo: (callTypeId: 3430040695729964317))))
; CHECK-DAG: (name: "device", {{.*}}, callTypeIdInfo: (callTypeId: 5970146889962867861))))
; CHECK-DAG: (name: "gfx", {{.*}}, callTypeIdInfo: (callTypeId: 11963068178503280946))))
; CHECK-DAG: (name: "resolver", {{.*}}, callTypeIdInfo: (callTypeId: 798791889831736034), refs: (^{{[0-9]+}}))))
;; A local function has a call type ID only if its address is taken. Without
;; one, the summary still has call type IDs, all empty.
; CHECK-DAG: (name: "local_not_taken", {{.*}}, callTypeIdInfo: (callTypeId: 0))))
; CHECK-DAG: (name: "local_taken", {{.*}}, callTypeIdInfo: (callTypeId: 5485679292981660066))))
;; A kernel cannot be called, so it has no call type ID. Its indirect calls
;; have distinct IDs, and its call to inline assembly has none.
; CHECK-DAG: (name: "kernel", {{.*}}, calls: ((callee: ^{{[0-9]+}})), callTypeIdInfo: (callTypeId: 0, indirectCallTypeIds: (3430040695729964317, 11963068178503280946), hasCallWithoutTypeId: 1))))
;; A call to an ifunc, also through an alias, has the call type ID of its type.
; CHECK-DAG: (name: "ifunc_caller", {{.*}}, callTypeIdInfo: (callTypeId: 5485679292981660066, indirectCallTypeIds: (3430040695729964317)))))
;; Inline assembly in callbr and invoke has no call type ID either.
; CHECK-DAG: (name: "asm_goto", {{.*}}, callTypeIdInfo: (callTypeId: 5485679292981660066, hasCallWithoutTypeId: 1))))
; CHECK-DAG: (name: "asm_invoke", {{.*}}, callTypeIdInfo: (callTypeId: 5485679292981660066, hasCallWithoutTypeId: 1), refs: (^{{[0-9]+}}))))
;; A call to an absolute address has no call type ID. Calls through
;; dso_local_equivalent and no_cfi values have one.
; CHECK-DAG: (name: "const_callers", {{.*}}, callTypeIdInfo: (callTypeId: 5485679292981660066, indirectCallTypeIds: (5485679292981660066, 5970146889962867861), hasCallWithoutTypeId: 1), refs: (^{{[0-9]+}}, ^{{[0-9]+}}))))

;; The bitcode analyzer prints the operands as signed values. The second operand
;; holds the flags.
; BCAN: <CALL_TYPE_IDS op0=0 op1=1 op2=3430040695729964317 op3=-6483675895206270670/>
; BCAN: <CALL_TYPE_IDS op0=5485679292981660066 op1=1 op2=5485679292981660066 op3=5970146889962867861/>

;; The combined index has no names. The GUIDs below are those of f_float,
;; kernel, and const_callers. Only local_not_taken has empty call type IDs.
; COMBINED-DAG: (guid: 6286836387820926777, {{.*}}, callTypeIdInfo: (callTypeId: 3430040695729964317))))
; COMBINED-DAG: insts: 1, callTypeIdInfo: (callTypeId: 0))))
; COMBINED-DAG: (guid: 17571550346185820240, {{.*}}, callTypeIdInfo: (callTypeId: 0, indirectCallTypeIds: (3430040695729964317, 11963068178503280946), hasCallWithoutTypeId: 1))))
; COMBINED-DAG: (guid: 14653077686872413374, {{.*}}, callTypeIdInfo: (callTypeId: 5485679292981660066, indirectCallTypeIds: (5485679292981660066, 5970146889962867861), hasCallWithoutTypeId: 1), refs: (^{{[0-9]+}}, ^{{[0-9]+}}))))

; NONE: (name: "kernel", {{.*}}, calls: ((callee: ^{{[0-9]+}})))))

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

@fp = global ptr @local_taken

@ifn = ifunc i32 (i32), ptr @resolver
@ifn_alias = alias i32 (i32), ptr @ifn

define float @f_float(float %x) {
  ret float %x
}

define i32 @f_i32(i32 %x) {
  ret i32 %x
}

define void @device(ptr %p) {
  ret void
}

define amdgpu_gfx void @gfx(ptr %p) {
  ret void
}

define internal void @local_not_taken() {
  ret void
}

define internal void @local_taken() {
  ret void
}

define amdgpu_kernel void @kernel(ptr %fp) {
  call void @local_not_taken()
  %a = call float %fp(float 1.0)
  %b = call i32 %fp(i32 1)
  call amdgpu_gfx void %fp(ptr null)
  call void asm sideeffect "", ""()
  ret void
}

define ptr @resolver() {
  ret ptr @f_i32
}

define void @ifunc_caller() {
  %a = call i32 @ifn(i32 1)
  %b = call i32 @ifn_alias(i32 2)
  ret void
}

define void @asm_goto() {
  callbr void asm "", "!i"()
          to label %normal [label %indirect]
normal:
  ret void
indirect:
  ret void
}

define void @asm_invoke() personality ptr @personality {
  invoke void asm sideeffect unwind "", ""()
          to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %lp = landingpad { ptr, i32 } cleanup
  ret void
}

declare i32 @personality(...)

define void @const_callers() {
  call void inttoptr (i64 4096 to ptr)()
  call void dso_local_equivalent @local_taken()
  call void no_cfi @device(ptr null)
  ret void
}
