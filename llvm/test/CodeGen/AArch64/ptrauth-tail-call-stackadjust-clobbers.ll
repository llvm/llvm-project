; RUN: llc -mtriple=arm64e-apple-ios < %s | FileCheck %s
; RUN: llc -mtriple=arm64e-apple-ios -global-isel -global-isel-abort=1 < %s | FileCheck %s

declare void @g()
declare i64 @llvm.ptrauth.auth(i64, i32 immarg, i64)

; The epilogue reconstructs the entry SP in x16 to authenticate LR when the
; tail call needs an SP adjustment, so the indirect call target must not live
; in x16 (or x15/x17, which other authentication forms use).
define swifttailcc void @f(ptr swiftasync %ctx, ptr %resume, i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 %a5, i64 %a6, i64 %a7, i64 %a8, i64 %a9) "frame-pointer"="non-leaf" "ptrauth-returns" "ptrauth-calls" "ptrauth-auth-traps" {
; CHECK-LABEL: f:
; CHECK:         sub x16, sp, #16
; CHECK-NEXT:    autib x30, x16
; CHECK-NOT:     br x1[567]
; CHECK:         br x{{([0-9]|1[0-4])$}}
  call void @g()
  %p = ptrtoint ptr %resume to i64
  %q = call i64 @llvm.ptrauth.auth(i64 %p, i32 0, i64 1234)
  %t = inttoptr i64 %q to ptr
  musttail call swifttailcc void %t(ptr swiftasync %ctx, i64 %a9)
  ret void
}

@fnptr = global ptr null
@disc = global i64 0
declare i64 @llvm.ptrauth.blend(i64, i64)

; Likewise for authenticated tail calls, for both the call target
; and the address discriminator. The inline asm leaves x15 as the only
; caller-saved register that can hold the value live across it.
define swifttailcc void @auth_dst(ptr swiftasync %ctx, ptr %resume, i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 %a5, i64 %a6, i64 %a7, i64 %a8, i64 %a9) "frame-pointer"="non-leaf" "ptrauth-returns" "branch-protection-pauth-lr" "ptrauth-calls" "target-features"="+pauth-lr" {
; CHECK-LABEL: auth_dst:
; CHECK:         autib171615
; CHECK-NOT:     x15
; CHECK:         mov x16, x{{([0-9]|1[0-4])$}}
; CHECK-NEXT:    movk x16, #42, lsl #48
; CHECK-NEXT:    braa x{{([0-9]|1[0-4])}}, x16
  call void @g()
  %fn = load volatile ptr, ptr @fnptr
  call void asm sideeffect "", "~{x0},~{x1},~{x2},~{x3},~{x4},~{x5},~{x6},~{x7},~{x8},~{x9},~{x10},~{x11},~{x12},~{x13},~{x14},~{x16},~{x17}"()
  %ad = load volatile i64, ptr @disc
  %b = call i64 @llvm.ptrauth.blend(i64 %ad, i64 42)
  musttail call swifttailcc void %fn(ptr swiftasync %ctx, i64 %a9) [ "ptrauth"(i32 0, i64 %b) ]
  ret void
}

define swifttailcc void @auth_addrdisc(ptr swiftasync %ctx, ptr %resume, i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 %a5, i64 %a6, i64 %a7, i64 %a8, i64 %a9) "frame-pointer"="non-leaf" "ptrauth-returns" "branch-protection-pauth-lr" "ptrauth-calls" "target-features"="+pauth-lr" {
; CHECK-LABEL: auth_addrdisc:
; CHECK:         autib171615
; CHECK-NOT:     x15
; CHECK:         mov x16, x{{([0-9]|1[0-4])$}}
; CHECK-NEXT:    movk x16, #42, lsl #48
; CHECK-NEXT:    braa x{{([0-9]|1[0-4])}}, x16
  call void @g()
  %ad = load volatile i64, ptr @disc
  call void asm sideeffect "", "~{x0},~{x1},~{x2},~{x3},~{x4},~{x5},~{x6},~{x7},~{x8},~{x9},~{x10},~{x11},~{x12},~{x13},~{x14},~{x16},~{x17}"()
  %fn = load volatile ptr, ptr @fnptr
  %b = call i64 @llvm.ptrauth.blend(i64 %ad, i64 42)
  musttail call swifttailcc void %fn(ptr swiftasync %ctx, i64 %a9) [ "ptrauth"(i32 0, i64 %b) ]
  ret void
}

; With BTI the call target must be x17, so the address discriminator must avoid x15 and x16.
define swifttailcc void @auth_bti_addrdisc(ptr swiftasync %ctx, i64 %a1) "frame-pointer"="non-leaf" "ptrauth-returns" "ptrauth-calls" "branch-target-enforcement" {
; CHECK-LABEL: auth_bti_addrdisc:
; CHECK:         mov x16, x{{([0-9]|1[0-4])$}}
; CHECK-NEXT:    movk x16, #42, lsl #48
; CHECK-NEXT:    braa x17, x16
  call void @g()
  %ad = load volatile i64, ptr @disc
  call void asm sideeffect "", "~{x0},~{x1},~{x2},~{x3},~{x4},~{x5},~{x6},~{x7},~{x8},~{x9},~{x10},~{x11},~{x12},~{x13},~{x14},~{x16},~{x17}"()
  %fn = load volatile ptr, ptr @fnptr
  %b = call i64 @llvm.ptrauth.blend(i64 %ad, i64 42)
  musttail call swifttailcc void %fn(ptr swiftasync %ctx, i64 %a1) [ "ptrauth"(i32 0, i64 %b) ]
  ret void
}
