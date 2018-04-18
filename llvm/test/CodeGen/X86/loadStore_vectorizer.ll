; RUN: opt -load-store-vectorizer < %s -S | FileCheck %s

%struct_render_pipeline_state = type opaque

define fastcc void @main(%struct_render_pipeline_state addrspace(1)* %pso) unnamed_addr {
; CHECK: load i16
; CHECK: load i16
entry:
  %tmp = bitcast %struct_render_pipeline_state addrspace(1)* %pso to i16 addrspace(1)*
  %tmp1 = load i16, i16 addrspace(1)* %tmp, align 2
  %tmp2 = bitcast %struct_render_pipeline_state addrspace(1)* %pso to i8 addrspace(1)*
  %sunkaddr51 = getelementptr i8, i8 addrspace(1)* %tmp2, i64 6
  %tmp3 = bitcast i8 addrspace(1)* %sunkaddr51 to i16 addrspace(1)*
  %tmp4 = load i16, i16 addrspace(1)* %tmp3, align 2
  ret void
}
