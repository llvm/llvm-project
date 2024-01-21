; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; TODO(#60133): Requires updates following opaque pointer migration.
; XFAIL: *
; REQUIRES: asserts

; CHECK: %[[#bool:]] = OpTypeBool
; CHECK: %[[#true:]] = OpConstantTrue %[[#bool]]
; CHECK: OpBranchConditional %[[#true]]

%structtype = type { i32, i32, i8 addrspace(4)* }
%structtype.0 = type <{ i32, i32, i8 addrspace(4)* }>

@__block_literal_global = internal addrspace(1) constant %structtype { i32 16, i32 8, i8 addrspace(4)* addrspacecast (i8* null to i8 addrspace(4)*) }, align 8
@__block_literal_global.1 = internal addrspace(1) constant %structtype { i32 16, i32 8, i8 addrspace(4)* addrspacecast (i8* null to i8 addrspace(4)*) }, align 8
@__block_literal_global.2 = internal addrspace(1) constant %structtype { i32 16, i32 8, i8 addrspace(4)* addrspacecast (i8* null to i8 addrspace(4)*) }, align 8

define spir_kernel void @block_typedef_mltpl_stmnt(i32 addrspace(1)* %res) {
entry:
  %0 = call spir_func <3 x i64> @BuiltInGlobalInvocationId()
  %call = extractelement <3 x i64> %0, i32 0
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %res, i64 %call
  store i32 -1, i32 addrspace(1)* %arrayidx, align 4
  %1 = bitcast %structtype addrspace(1)* @__block_literal_global to i8 addrspace(1)*
  %2 = addrspacecast i8 addrspace(1)* %1 to i8 addrspace(4)*
  %3 = bitcast %structtype addrspace(1)* @__block_literal_global.1 to i8 addrspace(1)*
  %4 = addrspacecast i8 addrspace(1)* %3 to i8 addrspace(4)*
  %5 = bitcast %structtype addrspace(1)* @__block_literal_global.2 to i8 addrspace(1)*
  %6 = addrspacecast i8 addrspace(1)* %5 to i8 addrspace(4)*
  br label %do.body

do.body:                                          ; preds = %do.cond, %entry
  %a.0 = phi i32 [ undef, %entry ], [ %a.1, %do.cond ]
  %call1 = call spir_func float @__block_typedef_mltpl_stmnt_block_invoke(i8 addrspace(4)* %2, float 0.000000e+00)
  %call2 = call spir_func i32 @__block_typedef_mltpl_stmnt_block_invoke_2(i8 addrspace(4)* %4, i32 0)
  %conv = sitofp i32 %call2 to float
  %sub = fsub float %call1, %conv
  %cmp = fcmp ogt float %sub, 0.000000e+00
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %do.body
  %call4 = call spir_func i32 @__block_typedef_mltpl_stmnt_block_invoke_3(i8 addrspace(4)* %6, i32 1)
  %call5 = call spir_func i32 @__block_typedef_mltpl_stmnt_block_invoke_3(i8 addrspace(4)* %6, i32 2)
  %add = add i32 %call4, %call5
  br label %cleanup

if.end:                                           ; preds = %do.body
  br label %cleanup

cleanup:                                          ; preds = %if.end, %if.then
  %a.1 = phi i32 [ %add, %if.then ], [ %a.0, %if.end ]
  %cleanup.dest.slot.0 = phi i32 [ 2, %if.then ], [ 0, %if.end ]
  switch i32 %cleanup.dest.slot.0, label %unreachable [
    i32 0, label %cleanup.cont
    i32 2, label %do.end
  ]

cleanup.cont:                                     ; preds = %cleanup
  br label %do.cond

do.cond:                                          ; preds = %cleanup.cont
  br i1 true, label %do.body, label %do.end

do.end:                                           ; preds = %do.cond, %cleanup
  %sub7 = sub nsw i32 %a.1, 11
  %arrayidx8 = getelementptr inbounds i32, i32 addrspace(1)* %res, i64 %call
  store i32 %sub7, i32 addrspace(1)* %arrayidx8, align 4
  ret void

unreachable:                                      ; preds = %cleanup
  unreachable
}

define internal spir_func float @__block_typedef_mltpl_stmnt_block_invoke(i8 addrspace(4)* %.block_descriptor, float %bi) {
entry:
  %block = bitcast i8 addrspace(4)* %.block_descriptor to %structtype.0 addrspace(4)*
  %conv = fpext float %bi to double
  %add = fadd double %conv, 3.300000e+00
  %conv1 = fptrunc double %add to float
  ret float %conv1
}

define internal spir_func i32 @__block_typedef_mltpl_stmnt_block_invoke_2(i8 addrspace(4)* %.block_descriptor, i32 %bi) {
entry:
  %block = bitcast i8 addrspace(4)* %.block_descriptor to %structtype.0 addrspace(4)*
  %add = add nsw i32 %bi, 2
  ret i32 %add
}

define internal spir_func i32 @__block_typedef_mltpl_stmnt_block_invoke_3(i8 addrspace(4)* %.block_descriptor, i32 %bi) {
entry:
  %block = bitcast i8 addrspace(4)* %.block_descriptor to %structtype.0 addrspace(4)*
  %add = add i32 %bi, 4
  ret i32 %add
}

declare spir_func <3 x i64> @BuiltInGlobalInvocationId()
