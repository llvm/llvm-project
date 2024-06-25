; RUN: llc < %s -march=nvptx -mcpu=sm_70 -o - 2>&1  | FileCheck %s

target triple = "nvptx64-nvidia-cuda"

@value = internal addrspace(1) global i128 0, align 16

; Function Attrs: alwaysinline convergent mustprogress willreturn
define void @_Z7kernel1v() {
  ; CHECK-LABEL: _Z7kernel1v
  ; CHECK: mov.u64    [[REG_HI:%rd[0-9]+]], 0;
  ; CHECK: mov.u64    [[REG_LO:%rd[0-9]+]], 42;
  ; CHECK: mov.b128   [[REG_128:%rq[0-9]+]], {[[REG_LO]], [[REG_HI]]};
  ; CHECK: { st.b128  [{{%rd[0-9]+}}], [[REG_128]]; }

  tail call void asm sideeffect "{ st.b128 [$0], $1; }", "l,q"(ptr nonnull addrspacecast (ptr addrspace(1) @value to ptr), i128 42)
  ret void
}

; Function Attrs: alwaysinline convergent mustprogress willreturn
define void @_Z7kernel2Pn(ptr nocapture readonly %data) {
  ; CHECK-LABEL: _Z7kernel2Pn
  ; CHECK: ld.global.u64 [[REG_HI:%rd[0-9]+]], [[[REG_Addr:%r[0-9]+]]+8];
  ; CHECK: ld.global.u64 [[REG_LO:%rd[0-9]+]], [[[REG_Addr]]];
  ; CHECK: mov.b128   [[REG_128:%rq[0-9]+]], {[[REG_LO]], [[REG_HI]]};
  ; CHECK: { st.b128  [{{%rd[0-9]+}}], [[REG_128]]; }

  %1 = addrspacecast ptr %data to ptr addrspace(1)
  %2 = load <2 x i64>, ptr addrspace(1) %1, align 16
  %3 = bitcast <2 x i64> %2 to i128
  tail call void asm sideeffect "{ st.b128 [$0], $1; }", "l,q"(ptr nonnull addrspacecast (ptr addrspace(1) @value to ptr), i128 %3)
  ret void
}

; Function Attrs: alwaysinline convergent mustprogress willreturn
define void @_Z7kernel3Pb(ptr nocapture readonly %flag) {
  ; CHECK-LABEL: _Z7kernel3Pb
  ; CHECK: selp.b64   [[REG_LO:%rd[0-9]+]], 24, 42, {{%p[0-9]+}};
  ; CHECK: mov.u64    [[REG_HI:%rd[0-9]+]], 0;
  ; CHECK: mov.b128   [[REG_128:%rq[0-9]+]], {[[REG_LO]], [[REG_HI]]};
  ; CHECK: { st.b128  [{{%rd[0-9]+}}], [[REG_128]]; }

  %1 = addrspacecast ptr %flag to ptr addrspace(1)
  %tmp1 = load i8, ptr addrspace(1) %1, align 1
  %tobool.not = icmp eq i8 %tmp1, 0
  %. = select i1 %tobool.not, i128 24, i128 42
  tail call void asm sideeffect "{ st.b128 [$0], $1; }", "l,q"(ptr nonnull addrspacecast (ptr addrspace(1) @value to ptr), i128 %.)
  ret void
}

; Function Attrs: alwaysinline mustprogress willreturn memory(write, argmem: none, inaccessiblemem: none)
define void @_Z7kernel4v() {
  ; CHECK-LABEL: _Z7kernel4v
  ; CHECK-O3: { mov.b128 [[REG_128:%rq[0-9]+]], 41; }
  ; CHECK-O3: mov.b128   {%rd{{[0-9]+}}, %rd{{[0-9]+}}}, [[REG_128]];
  
  %1 = tail call i128 asm "{ mov.b128 $0, 41; }", "=q"()
  %add = add nsw i128 %1, 1
  %2 = bitcast i128 %add to <2 x i64>
  store <2 x i64> %2, ptr addrspace(1) @value, align 16
  ret void
}

; Function Attrs: alwaysinline mustprogress willreturn memory(write, argmem: read, inaccessiblemem: none)
define void @_Z7kernel5Pn(ptr nocapture readonly %data) {
  ; CHECK-LABEL: _Z7kernel5Pn
  ; CHECK-O3: ld.global.v2.u64 {[[REG_LO_IN:%rd[0-9]+]], [[REG_HI_IN:%rd[0-9]+]]}, [{{%rd[0-9]+}}];
  ; CHECK-O3: mov.b128   [[REG_128_IN:%rq[0-9]+]], {[[REG_LO_IN]], [[REG_HI_IN]]};
  ; CHECK-O3: { mov.b128 [[REG_128_OUT:%rq[0-9]+]], [[REG_128_IN]]; }
  ; CHECK-O3: mov.b128   {%rd{{[0-9]+}}, %rd{{[0-9]+}}}, [[REG_128_OUT]];

  %1 = addrspacecast ptr %data to ptr addrspace(1)
  %2 = load <2 x i64>, ptr addrspace(1) %1, align 16
  %3 = bitcast <2 x i64> %2 to i128
  %4 = tail call i128 asm "{ mov.b128 $0, $1; }", "=q,q"(i128 %3)
  %add = add nsw i128 %4, 1
  %5 = bitcast i128 %add to <2 x i64>
  store <2 x i64> %5, ptr addrspace(1) @value, align 16
  ret void
}

!nvvmir.version = !{!0, !1, !0, !1, !1, !0, !0, !0, !1}

!0 = !{i32 2, i32 0, i32 3, i32 1}
!1 = !{i32 2, i32 0}
