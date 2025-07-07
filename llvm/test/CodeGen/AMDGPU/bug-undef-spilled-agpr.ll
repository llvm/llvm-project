; Just ensure that llc -O1 does not error out
; RUN: llc -O1 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -verify-machineinstrs %s -o - &>/dev/null

define fastcc void @widget(i1 %arg) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb3, %bb
  br i1 %arg, label %bb3, label %bb2

bb2:                                              ; preds = %bb1
  ret void

bb3:                                              ; preds = %bb1
  %call = call fastcc i1 @baz(i1 false, float 0.000000e+00, i1 false, float 0.000000e+00, i1 false, i1 false, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, ptr addrspace(5) null, i1 false, ptr null, ptr null, ptr null, ptr null, ptr null, ptr addrspace(5) null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr addrspace(5) null)
  br label %bb1
}

define fastcc i1 @baz(i1 %arg, float %arg1, i1 %arg2, float %arg3, i1 %arg4, i1 %arg5, float %arg6, float %arg7, float %arg8, float %arg9, ptr addrspace(5) %arg10, i1 %arg11, ptr %arg12, ptr %arg13, ptr %arg14, ptr %arg15, ptr %arg16, ptr addrspace(5) %arg17, ptr %arg18, ptr %arg19, ptr %arg20, ptr %arg21, ptr %arg22, ptr %arg23, ptr %arg24, ptr addrspace(5) %arg25) #0 {
bb:
  br i1 %arg, label %bb26, label %bb27

bb26:                                             ; preds = %bb
  ret i1 false

bb27:                                             ; preds = %bb
  br i1 %arg, label %bb29, label %bb28

bb28:                                             ; preds = %bb27
  unreachable

bb29:                                             ; preds = %bb49, %bb47, %bb46, %bb39, %bb36, %bb27
  br i1 %arg4, label %bb55, label %bb30

bb30:                                             ; preds = %bb29
  br i1 %arg5, label %bb31, label %bb32

bb31:                                             ; preds = %bb30
  store i1 false, ptr addrspace(5) %arg17, align 8
  br label %bb55

bb32:                                             ; preds = %bb30
  store float %arg3, ptr addrspace(5) %arg25, align 8
  store float %arg7, ptr addrspace(5) %arg10, align 8
  br i1 %arg2, label %bb34, label %bb33

bb33:                                             ; preds = %bb32
  %fcmp = fcmp ogt float %arg6, 0.000000e+00
  br i1 %fcmp, label %bb34, label %bb35

bb34:                                             ; preds = %bb33, %bb32
  br i1 %arg11, label %bb37, label %bb36

bb35:                                             ; preds = %bb33
  store float 0.000000e+00, ptr addrspace(5) %arg25, align 8
  ret i1 false

bb36:                                             ; preds = %bb34
  store i32 1, ptr addrspace(5) %arg17, align 8
  br label %bb29

bb37:                                             ; preds = %bb34
  %load = load i8, ptr %arg12, align 2
  %trunc = trunc i8 %load to i1
  br i1 %trunc, label %bb38, label %bb54

bb38:                                             ; preds = %bb37
  br i1 %arg4, label %bb39, label %bb53

bb39:                                             ; preds = %bb38
  store float %arg1, ptr addrspace(5) %arg25, align 8
  %load40 = load float, ptr %arg15, align 8
  call void @llvm.memcpy.p5.p0.i64(ptr addrspace(5) %arg25, ptr %arg24, i64 12, i1 false)
  %load41 = load float, ptr %arg16, align 4
  call void @llvm.memcpy.p5.p0.i64(ptr addrspace(5) %arg17, ptr null, i64 36, i1 false)
  %load42 = load float, ptr %arg18, align 4
  %load43 = load float, ptr %arg19, align 4
  store float 0x7FF8000000000000, ptr addrspace(5) %arg25, align 8
  %load44 = load float, ptr %arg14, align 16
  store float %load44, ptr addrspace(5) %arg25, align 8
  %fcmp45 = fcmp ole float %arg9, 0.000000e+00
  br i1 %fcmp45, label %bb29, label %bb46

bb46:                                             ; preds = %bb39
  %fsub = fsub float %arg8, %load40
  store float %fsub, ptr addrspace(5) %arg25, align 8
  %fadd = fadd float %load42, %load43
  br i1 %arg, label %bb29, label %bb47

bb47:                                             ; preds = %bb46
  br i1 %arg, label %bb29, label %bb48

bb48:                                             ; preds = %bb47
  br i1 %arg, label %bb49, label %bb52

bb49:                                             ; preds = %bb48
  store float 0.000000e+00, ptr %arg23, align 4
  store float 0.000000e+00, ptr %arg22, align 8
  store float %fadd, ptr addrspace(5) %arg25, align 8
  %load50 = load float, ptr %arg20, align 4
  %fdiv = fdiv float %load41, %load50
  store float %fdiv, ptr addrspace(5) %arg25, align 8
  %load51 = load float, ptr %arg13, align 16
  store float %load51, ptr addrspace(5) %arg25, align 8
  store float 1.000000e+00, ptr %arg21, align 4
  br label %bb29

bb52:                                             ; preds = %bb48
  unreachable

bb53:                                             ; preds = %bb38
  ret i1 false

bb54:                                             ; preds = %bb37
  ret i1 true

bb55:                                             ; preds = %bb31, %bb29
  %load56 = load i1, ptr addrspace(5) %arg25, align 8
  ret i1 %load56
}

declare void @llvm.memcpy.p5.p0.i64(ptr addrspace(5) noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1

attributes #0 = { "denormal-fp-math-f32"="preserve-sign,preserve-sign" }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
