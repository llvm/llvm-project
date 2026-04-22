; Verify that the codegen-only CAS*/SWP*/LDAPR* instructions emitted by the MS
; __cas*/__swp*/__ldapr* intrinsics can be re-assembled when no -march feature
; enables LSE/RCPC. The AsmPrinter brackets each such instruction with
; .arch_extension directives so the integrated assembler (-S then assemble, or
; -save-temps) accepts the otherwise feature-gated encoding.

; RUN: llc -mtriple=aarch64-windows-msvc < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-windows-msvc < %s | \
; RUN:     llvm-mc -triple=aarch64-windows-msvc -filetype=obj -o /dev/null

define i32 @cas(ptr %p, i32 %rs, i32 %rt) {
; CHECK-LABEL: cas:
; CHECK:         .arch_extension lse
; CHECK-NEXT:    cas w1, w2, [x0]
; CHECK-NEXT:    .arch_extension nolse
  %r = call i32 @llvm.aarch64.cas32(ptr %p, i32 %rs, i32 %rt)
  ret i32 %r
}

define i32 @swp(ptr %p, i32 %rs) {
; CHECK-LABEL: swp:
; CHECK:         .arch_extension lse
; CHECK-NEXT:    swp w1, w0, [x0]
; CHECK-NEXT:    .arch_extension nolse
  %r = call i32 @llvm.aarch64.swp32(ptr %p, i32 %rs)
  ret i32 %r
}

define i32 @ldapr(ptr %p) {
; CHECK-LABEL: ldapr:
; CHECK:         .arch_extension rcpc
; CHECK-NEXT:    ldapr w0, [x0]
; CHECK-NEXT:    .arch_extension norcpc
  %r = call i32 @llvm.aarch64.ldapr32(ptr %p)
  ret i32 %r
}
