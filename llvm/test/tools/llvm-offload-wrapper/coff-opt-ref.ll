; REQUIRES: x86-registered-target
; REQUIRES: lld

; RUN: rm -rf %t && split-file %s %t
; RUN: llvm-offload-wrapper --triple=x86_64-pc-windows-msvc -kind=hip \
; RUN:   %t/image.bin -o %t/wrapper.bc
; RUN: llvm-dis %t/wrapper.bc -o - | FileCheck %s
; RUN: llc -filetype=obj %t/wrapper.bc -o %t/wrapper.obj
; RUN: llvm-readobj --sections %t/wrapper.obj | FileCheck %s --check-prefix=OBJ
; RUN: llc -filetype=obj %t/stubs.ll -o %t/stubs.obj
; RUN: lld-link /dll /noentry /nodefaultlib /opt:ref /map:%t/wrapper.map \
; RUN:   /out:%t/wrapper.dll %t/wrapper.obj %t/stubs.obj
; RUN: FileCheck %s --check-prefix=MAP --input-file=%t/wrapper.map

; CHECK: @__start_llvm_offload_entries = weak_odr hidden constant [1 x %struct.__tgt_offload_entry] zeroinitializer, section "llvm_offload_entries$OA"
; CHECK-NEXT: @__stop_llvm_offload_entries = weak_odr hidden constant [1 x %struct.__tgt_offload_entry] zeroinitializer, section "llvm_offload_entries$OZ"
; CHECK: icmp ne ptr getelementptr inbounds ([1 x %struct.__tgt_offload_entry], ptr @__start_llvm_offload_entries, i32 0, i32 1), @__stop_llvm_offload_entries

; OBJ: Name: llvm_offload_entries{{[$]}}OA
; OBJ: RawDataSize: 56
; OBJ: IMAGE_SCN_ALIGN_8BYTES
; OBJ: Name: llvm_offload_entries{{[$]}}OZ
; OBJ: RawDataSize: 56
; OBJ: IMAGE_SCN_ALIGN_8BYTES

; MAP:      {{[0-9A-Fa-f]+}}:00000000 00000038H llvm_offload_entries{{[$]}}OA DATA
; MAP-NEXT: {{[0-9A-Fa-f]+}}:00000038 00000038H llvm_offload_entries{{[$]}}OZ DATA
; MAP:      {{[0-9A-Fa-f]+}}:00000000 __start_llvm_offload_entries
; MAP:      {{[0-9A-Fa-f]+}}:00000038 __stop_llvm_offload_entries

;--- image.bin
device image

;--- stubs.ll
target triple = "x86_64-pc-windows-msvc"

define ptr @__hipRegisterFatBinary(ptr %x) {
entry:
  ret ptr %x
}

define void @__cudaRegisterFatBinaryEnd(ptr %x) {
entry:
  ret void
}

define void @__hipUnregisterFatBinary(ptr %x) {
entry:
  ret void
}

define i32 @atexit(ptr %f) {
entry:
  ret i32 0
}

define i32 @__hipRegisterFunction(ptr %handle, ptr %addr, ptr %name,
                                  ptr %name2, i32 %thread_limit, ptr %tid,
                                  ptr %bid, ptr %bDim, ptr %gDim,
                                  ptr %wSize) {
entry:
  ret i32 0
}

define void @__hipRegisterVar(ptr %handle, ptr %addr, ptr %name, ptr %name2,
                              i32 %ext, i64 %size, i32 %constant,
                              i32 %global) {
entry:
  ret void
}

define void @__hipRegisterManagedVar(ptr %handle, ptr %aux, ptr %addr,
                                     ptr %name, i64 %size, i32 %flags) {
entry:
  ret void
}

define void @__hipRegisterSurface(ptr %handle, ptr %addr, ptr %name,
                                  ptr %name2, i32 %dim, i32 %ext) {
entry:
  ret void
}

define void @__hipRegisterTexture(ptr %handle, ptr %addr, ptr %name,
                                  ptr %name2, i32 %dim, i32 %normalized,
                                  i32 %ext) {
entry:
  ret void
}
