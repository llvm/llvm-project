; RUN: opt -verify-each -passes='coro-early' -S %s | FileCheck %s
; RUN: opt -verify-each -passes='coro-early,cgscc(coro-split),coro-cleanup' -disable-output %s

target triple = "x86_64-pc-windows-msvc"
$exported = comdat any

declare token @llvm.coro.id.retcon.once(i32, i32, ptr, ptr, ptr, ptr, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.suspend.retcon.i1(...)
declare void @llvm.coro.end(ptr, i1, token)
declare void @llvm.trap()
declare void @prototype(ptr, i1)
declare ptr @malloc(i64)
declare void @free(ptr)

; The exported symbol retains its linkage, DLL storage, COMDAT and type
; metadata. The anonymous implementation must not acquire those contracts.
; CHECK-LABEL: define linkonce_odr dllexport { ptr, i32 } @exported(ptr nonnull %buffer) {{.*}}comdat !type
; CHECK: call ptr @llvm.coro.prepare.retcon(ptr @exported.coro.body)
; CHECK: musttail call { ptr, i32 } {{%.*}}(ptr nonnull %buffer)
; CHECK-LABEL: define internal { ptr, i32 } @exported.coro.body(ptr nonnull %buffer)
; CHECK-SAME: #[[BODY:[0-9]+]] {
define linkonce_odr dllexport { ptr, i32 } @exported(ptr nonnull %buffer) noinline optnone presplitcoroutine comdat !type !0 {
entry:
  %return.slot = alloca { ptr, i32 }, align 8
  %id = call token @llvm.coro.id.retcon.once(i32 16, i32 8, ptr %buffer,
      ptr @prototype, ptr @malloc, ptr @free, ptr %return.slot)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  %unwind = call i1 (...) @llvm.coro.suspend.retcon.i1(i32 7)
  call void @llvm.trap()
  unreachable

cleanup:
  call void @llvm.coro.end(ptr %hdl, i1 false, token none)
  %result = load { ptr, i32 }, ptr %return.slot, align 8
  ret { ptr, i32 } %result
}

; A musttail forwarding entry preserves variadic arguments as well.
; CHECK-LABEL: define { ptr, i32 } @variadic(ptr %buffer, ...)
; CHECK: [[PREPARED:%.*]] = call ptr @llvm.coro.prepare.retcon(ptr @variadic.coro.body)
; CHECK-NEXT: [[RESULT:%.*]] = musttail call { ptr, i32 } (ptr, ...) [[PREPARED]](ptr %buffer, ...)
; CHECK-NEXT: ret { ptr, i32 } [[RESULT]]
define { ptr, i32 } @variadic(ptr %buffer, ...) presplitcoroutine {
entry:
  %return.slot = alloca { ptr, i32 }, align 8
  %id = call token @llvm.coro.id.retcon.once(i32 16, i32 8, ptr %buffer,
      ptr @prototype, ptr @malloc, ptr @free, ptr %return.slot)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  %unwind = call i1 (...) @llvm.coro.suspend.retcon.i1(i32 7)
  call void @llvm.trap()
  unreachable

cleanup:
  call void @llvm.coro.end(ptr %hdl, i1 false, token none)
  %result = load { ptr, i32 }, ptr %return.slot, align 8
  ret { ptr, i32 } %result
}

; Swift's calling convention and ABI parameter attributes belong on the
; forwarding call as well as on the public symbol.
; CHECK-LABEL: define swiftcc { ptr, i32 } @swift_entry(ptr swiftself %buffer)
; CHECK: musttail call swiftcc { ptr, i32 } {{%.*}}(ptr swiftself %buffer)
define swiftcc { ptr, i32 } @swift_entry(ptr swiftself %buffer) presplitcoroutine {
entry:
  %return.slot = alloca { ptr, i32 }, align 8
  %id = call token @llvm.coro.id.retcon.once(i32 16, i32 8, ptr %buffer,
      ptr @prototype, ptr @malloc, ptr @free, ptr %return.slot)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  %unwind = call i1 (...) @llvm.coro.suspend.retcon.i1(i32 7)
  call void @llvm.trap()
  unreachable

cleanup:
  call void @llvm.coro.end(ptr %hdl, i1 false, token none)
  %result = load { ptr, i32 }, ptr %return.slot, align 8
  ret { ptr, i32 } %result
}

; The prepare intrinsic uses address space zero; preserve the function's
; address space on the callable entry and the forwarded call.
; CHECK-LABEL: define { ptr addrspace(1), i32 } @address_space(ptr %buffer) addrspace(1)
; CHECK: call ptr @llvm.coro.prepare.retcon(ptr addrspacecast (ptr addrspace(1) @address_space.coro.body to ptr))
; CHECK: musttail call addrspace(1) { ptr addrspace(1), i32 }
define { ptr addrspace(1), i32 } @address_space(ptr %buffer) addrspace(1) presplitcoroutine {
entry:
  %return.slot = alloca { ptr addrspace(1), i32 }, align 8
  %id = call token @llvm.coro.id.retcon.once(i32 16, i32 8, ptr %buffer,
      ptr @prototype, ptr @malloc, ptr @free, ptr %return.slot)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  %unwind = call i1 (...) @llvm.coro.suspend.retcon.i1(i32 7)
  call void @llvm.trap()
  unreachable

cleanup:
  call void @llvm.coro.end(ptr %hdl, i1 false, token none)
  %result = load { ptr addrspace(1), i32 }, ptr %return.slot, align 8
  ret { ptr addrspace(1), i32 } %result
}

; CHECK: attributes #[[BODY]] = { noinline optnone presplitcoroutine "coro-retcon-body" }
!0 = !{i64 0, !"entry-type"}
