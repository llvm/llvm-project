; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s
;
; When !ptrauth.resume metadata is attached to coro.id.retcon.once,
; CoroSplit signs the resume pointer using llvm.ptrauth.sign.

target triple = "arm64e-apple-darwin"

declare void @prototype(ptr, i1)
declare noalias ptr @allocate(i32)
declare void @deallocate(ptr)

; Test address-diversified signing (addr_div=true).
; The discriminator is blended with the buffer address before signing.
define {ptr, ptr} @test_ptrauth_resume(ptr %buffer, ptr %ptr) presplitcoroutine {
entry:
  %temp = alloca i32, align 4
  %id = call token @llvm.coro.id.retcon.once(i32 8, i32 8, ptr %buffer, ptr @prototype, ptr @allocate, ptr @deallocate), !ptrauth.resume !0
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  %oldvalue = load i32, ptr %ptr
  store i32 %oldvalue, ptr %temp
  %unwind = call i1 (...) @llvm.coro.suspend.retcon.i1(ptr %temp)
  br i1 %unwind, label %cleanup, label %cont

cont:
  %newvalue = load i32, ptr %temp
  store i32 %newvalue, ptr %ptr
  br label %cleanup

cleanup:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  unreachable
}

; CHECK-LABEL: define { ptr, ptr } @test_ptrauth_resume(
; Address-diversified: blend buffer address with discriminator, then sign.
; CHECK: %[[BUFINT:.*]] = ptrtoint ptr %buffer to i64
; CHECK: %[[BLEND:.*]] = call i64 @llvm.ptrauth.blend(i64 %[[BUFINT]], i64 3909)
; CHECK: %[[SIGNED:.*]] = call i64 @llvm.ptrauth.sign(i64 ptrtoint (ptr @test_ptrauth_resume.resume.0 to i64), i32 0, i64 %[[BLEND]])
; CHECK: %[[PTR:.*]] = inttoptr i64 %[[SIGNED]] to ptr
; CHECK: ret { ptr, ptr }

; Test non-address-diversified signing (addr_div=false).
; The discriminator is used directly without blending.
define {ptr, ptr} @test_ptrauth_resume_no_addrdiv(ptr %buffer, ptr %ptr) presplitcoroutine {
entry:
  %temp = alloca i32, align 4
  %id = call token @llvm.coro.id.retcon.once(i32 8, i32 8, ptr %buffer, ptr @prototype, ptr @allocate, ptr @deallocate), !ptrauth.resume !1
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  %oldvalue = load i32, ptr %ptr
  store i32 %oldvalue, ptr %temp
  %unwind = call i1 (...) @llvm.coro.suspend.retcon.i1(ptr %temp)
  br i1 %unwind, label %cleanup, label %cont

cont:
  %newvalue = load i32, ptr %temp
  store i32 %newvalue, ptr %ptr
  br label %cleanup

cleanup:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  unreachable
}

; CHECK-LABEL: define { ptr, ptr } @test_ptrauth_resume_no_addrdiv(
; Non-address-diversified: no blend, discriminator passed directly to sign.
; CHECK-NOT: @llvm.ptrauth.blend
; CHECK: %[[SIGNED2:.*]] = call i64 @llvm.ptrauth.sign(i64 ptrtoint (ptr @test_ptrauth_resume_no_addrdiv.resume.0 to i64), i32 0, i64 3909)
; CHECK: %[[PTR2:.*]] = inttoptr i64 %[[SIGNED2]] to ptr
; CHECK: ret { ptr, ptr }

declare token @llvm.coro.id.retcon.once(i32, i32, ptr, ptr, ptr, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.suspend.retcon.i1(...)
declare void @llvm.coro.end(ptr, i1, token)

; key=0 (IA), disc=3909 (0x0F45), addr_div=true
!0 = !{i32 0, i64 3909, i1 true}
; key=0 (IA), disc=3909 (0x0F45), addr_div=false
!1 = !{i32 0, i64 3909, i1 false}
