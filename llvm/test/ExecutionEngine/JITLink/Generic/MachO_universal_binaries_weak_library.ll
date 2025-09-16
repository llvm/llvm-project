; RUN: rm -rf %t && mkdir %t
; RUN: yaml2obj -o %t/libFoo.dylib %S/Inputs/MachO_Universal_libFoo_dylib.yaml
; RUN: llc -filetype=obj -mtriple arm64e-apple-macosx -o %t/main.o %s
; RUN: llvm-jitlink -noexec -triple arm64e-apple-macosx %t/main.o -weak_library \
; RUN:     %t/libFoo.dylib
;
; REQUIRES: x86-registered-target && aarch64-registered-target
;
; Check MachO universal binary handling in the orc::getDylibInterfaceFromDylib
; function, including that the cpusubtype field is masked correctly (for arm64e
; slices this field will have the MachO::CPU_SUBTYPE_LIB64 flag set in the high
; bits -- the subtype will fail to match unless it's masked out).

declare i32 @foo()

define i32 @main(i32 %argc, ptr %argv) {
entry:
  ret i32 ptrtoint (ptr @foo to i32)
}
