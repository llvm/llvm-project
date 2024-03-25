; RUN: opt -S -passes=globalopt < %s | FileCheck %s
;; Check that Global Opt preserves address space of llvm.used and
;; llvm.compiler.used variables.

%struct.FakeDeviceGlobal = type { ptr addrspace(4) }
%class.anon = type { i8 }

@_ZM2C = internal addrspace(1) global %struct.FakeDeviceGlobal zeroinitializer, align 8
@_ZL1C = internal addrspace(1) global %struct.FakeDeviceGlobal zeroinitializer, align 8

@llvm.compiler.used = appending global [2 x ptr addrspace(4)] [ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZM2C to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZL1C to ptr addrspace(4))]

; CHECK: @llvm.compiler.used = appending global [2 x ptr addrspace(4)] [ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZL1C to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZM2C to ptr addrspace(4))]

define weak_odr dso_local void @foo() {
entry:
  %A = alloca %class.anon, align 1
  %A.addrspacecast = addrspacecast ptr %A to ptr addrspace(4)
  call void @bar(ptr addrspace(4) noundef align 1 dereferenceable_or_null(1) %A.addrspacecast)
  ret void
}

define internal void @bar(ptr addrspace(4) noundef align 1 dereferenceable_or_null(1) %this) align 2 {
entry:
  %this.addr = alloca ptr addrspace(4), align 8
  %this.addr.ascast = addrspacecast ptr %this.addr to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr addrspace(4) %this.addr.ascast, align 8
  %v1 = load ptr addrspace(4), ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZM2C to ptr addrspace(4)), align 8
  store i32 42, ptr addrspace(4) %v1, align 4
  %v2 = load ptr addrspace(4), ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZL1C to ptr addrspace(4)), align 8
  store i32 42, ptr addrspace(4) %v2, align 4
  ret void
}

