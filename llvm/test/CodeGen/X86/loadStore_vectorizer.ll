; RUN: opt -mtriple x86_64-- -load-store-vectorizer < %s -S | FileCheck %s

%struct_render_pipeline_state = type opaque

define fastcc void @test1(ptr addrspace(1) %pso) unnamed_addr {
; CHECK-LABEL: @test1
; CHECK: load i16
; CHECK: load i16
entry:
  %tmp1 = load i16, ptr addrspace(1) %pso, align 2
  %sunkaddr51 = getelementptr i8, ptr addrspace(1) %pso, i64 6
  %tmp4 = load i16, ptr addrspace(1) %sunkaddr51, align 2
  ret void
}

define fastcc void @test2(ptr addrspace(1) %pso) unnamed_addr {
; CHECK-LABEL: @test2
; CHECK: load <2 x i16>
entry:
  %tmp1 = load i16, ptr addrspace(1) %pso, align 2
  %sunkaddr51 = getelementptr i8, ptr addrspace(1) %pso, i64 2
  %tmp4 = load i16, ptr addrspace(1) %sunkaddr51, align 2
  ret void
}
