; RUN: llc -mcpu=diamondrapids %s -mtriple=x86_64 -filetype=obj -o %t.o
; RUN: llvm-objdump --no-print-imm-hex -dr %t.o | FileCheck %s --check-prefixes=NOAPXREL,CHECK

; RUN: llc -mcpu=diamondrapids %s -mtriple=x86_64 -filetype=obj -o %t.o -x86-enable-apx-for-relocation=true
; RUN: llvm-objdump --no-print-imm-hex -dr %t.o | FileCheck %s --check-prefixes=APXREL,CHECK


; All tests are used to check no R_X86_64_CODE_4_GOTPCRELX relocation type
; emitted if APX features is disabled for relocation.
; The first 2 tests are used to check if the register class is not
; updated/recomputed by register allocator. It's originally updated to non-rex2
; register class by "Suppress APX for relocation" pass.


; CHECK-LABEL: test_regclass_not_updated_by_regalloc_1
; CHECK-NOT: R_X86_64_CODE_4_GOTPCRELX gvar-0x4
; CHECK: movq    (%rip), %rdi
; CHECK-NEXT: R_X86_64_REX_GOTPCRELX gvar-0x4

@gvar = external global [20000 x i8]

define void @test_regclass_not_updated_by_regalloc_1(ptr %ptr1, ptr %0, i32 %int1, i64 %int_sext, i64 %mul.447, i64 %int_sext3, i32 %fetch.2508, i32 %fetch.2513, i32 %mul.442, i64 %int_sext6, i64 %int_sext7, i64 %int_sext8, i1 %cond1, i1 %cond2) {
alloca_38:
  %int_sext4 = sext i32 %int1 to i64
  tail call void @llvm.memset.p0.i64(ptr @gvar, i8 0, i64 20000, i1 false)
  %div.161 = sdiv i64 %int_sext3, %int_sext
  %cmp.2 = icmp sgt i64 %div.161, 0
  %1 = sub i64 %int_sext7, %mul.447
  br label %loop.41

loop.41:                                          ; preds = %ifmerge.2, %alloca_38
  br i1 %cmp.2, label %L.53, label %ifmerge.2

L.53:                                         ; preds = %loop.41
  %2 = getelementptr i8, ptr %ptr1, i64 %int_sext8
  br label %loop.83

loop.83:                                          ; preds = %loop.83, %L.53
  %i2.i64.1 = phi i64 [ 0, %L.53 ], [ %nextloop.83, %loop.83 ]
  %3 = mul i64 %i2.i64.1, %int_sext4
  %.r275 = add i64 %3, %1
  %4 = getelementptr float, ptr getelementptr ([20000 x i8], ptr @gvar, i64 0, i64 8000), i64 %.r275
  %gepload = load float, ptr %2, align 1
  store float %gepload, ptr %4, align 4
  %nextloop.83 = add i64 %i2.i64.1, 1
  br i1 %cond1, label %ifmerge.2, label %loop.83

ifmerge.2:                                        ; preds = %loop.83, %loop.41
  br i1 %cond2, label %afterloop.41, label %loop.41

afterloop.41:                                     ; preds = %ifmerge.2
  %mul.469 = mul i32 %mul.442, %fetch.2508
  %div.172 = mul i32 %fetch.2513, %mul.469
  %mul.471 = mul i32 %int1, %div.172
  %int_sext39 = sext i32 %mul.471 to i64
  %5 = mul i64 %int_sext6, %int_sext39
  %6 = getelementptr i8, ptr %ptr1, i64 %5
  %7 = load float, ptr %6, align 1
  store float %7, ptr null, align 4
  ret void
}

declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)

; TODO: update after R_X86_64_CODE_6_GOTPCRELX is supported.
; CHECK-LABEL: test_regclass_not_updated_by_regalloc_2
; APXREL: {nf} addq (%rip), %r16, %rcx
; APXREL-NEXT: R_X86_64_GOTPCREL gvar2-0x4
; NOAPXREL-NOT: R_X86_64_CODE_4_GOTPCRELX gvar2-0x4
; NOAPXREL: addq    (%rip), %rbx
; NOAPXREL-NEXT: R_X86_64_REX_GOTPCRELX gvar2-0x4

@gvar2 = external constant [8 x [8 x i32]]

define void @test_regclass_not_updated_by_regalloc_2(ptr %pSrc1, i32 %srcStep1, ptr %pSrc2, i32 %srcStep2, i32 %width, i32 %0, i1 %cmp71.not783, i1 %cmp11.i, ptr %pSrc2.addr.0535.i) {
entry:
  %1 = ashr i32 %srcStep2, 1
  %conv.i = sext i32 %width to i64
  %conv6.i = and i32 %srcStep1, 1
  %cmp.i = icmp sgt i32 %srcStep1, 0
  %idx.ext.i = zext i32 %conv6.i to i64
  %2 = getelementptr <4 x i64>, ptr @gvar2, i64 %idx.ext.i
  %idx.ext183.i = sext i32 %1 to i64
  br i1 %cmp71.not783, label %for.end, label %for.body73.lr.ph

for.body73.lr.ph:                                 ; preds = %entry
  %3 = load <4 x i64>, ptr %2, align 32
  %..i = select i1 %cmp11.i, <4 x i64> zeroinitializer, <4 x i64> splat (i64 1)
  %4 = bitcast <4 x i64> %..i to <8 x i32>
  %5 = bitcast <4 x i64> %3 to <8 x i32>
  %. = select i1 %cmp.i, <8 x i32> splat (i32 1), <8 x i32> %4
  %.833 = select i1 %cmp.i, <8 x i32> %5, <8 x i32> zeroinitializer
  br i1 %cmp11.i, label %for.end.i, label %for.end

for.end.i:                                        ; preds = %if.end153.i, %for.body73.lr.ph
  %pSrc2.addr.0535.i5 = phi ptr [ %add.ptr184.i, %if.end153.i ], [ %pSrc2, %for.body73.lr.ph ]
  %eSum0.0531.i = phi <4 x i64> [ %add.i452.i, %if.end153.i ], [ zeroinitializer, %for.body73.lr.ph ]
  br i1 %cmp71.not783, label %if.end153.i, label %if.then90.i

if.then90.i:                                      ; preds = %for.end.i
  %6 = tail call <8 x i32> @llvm.x86.avx2.maskload.d.256(ptr null, <8 x i32> %.)
  %add.i464.i = or <4 x i64> %eSum0.0531.i, zeroinitializer
  %7 = bitcast <8 x i32> %.833 to <4 x i64>
  %add.ptr152.i = getelementptr i16, ptr %pSrc2.addr.0535.i5, i64 %conv.i
  br label %if.end153.i

if.end153.i:                                      ; preds = %if.then90.i, %for.end.i
  %eSum0.2.i = phi <4 x i64> [ %7, %if.then90.i ], [ %eSum0.0531.i, %for.end.i ]
  %pLocSrc2.1.i = phi ptr [ %add.ptr152.i, %if.then90.i ], [ %pSrc1, %for.end.i ]
  %8 = load i16, ptr %pLocSrc2.1.i, align 2
  %conv165.i = zext i16 %8 to i32
  %vecinit3.i.i = insertelement <4 x i32> zeroinitializer, i32 %conv165.i, i64 0
  %9 = bitcast <4 x i32> %vecinit3.i.i to <2 x i64>
  %shuffle.i503.i = shufflevector <2 x i64> %9, <2 x i64> zeroinitializer, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %add.i452.i = or <4 x i64> %eSum0.2.i, %shuffle.i503.i
  %add.ptr184.i = getelementptr i16, ptr %pSrc2.addr.0535.i, i64 %idx.ext183.i
  br label %for.end.i

for.end:                                          ; preds = %for.body73.lr.ph, %entry
  br label %for.cond29.preheader.i227

for.cond29.preheader.i227:                        ; preds = %for.end
  br label %for.body32.i328

for.body32.i328:                                  ; preds = %for.body32.i328, %for.cond29.preheader.i227
  %w.0524.i329 = phi i32 [ %sub.i381, %for.body32.i328 ], [ 0, %for.cond29.preheader.i227 ]
  %sub.i381 = or i32 %w.0524.i329, 0
  %cmp30.i384 = icmp sgt i32 %w.0524.i329, 0
  br label %for.body32.i328
}

declare <8 x i32> @llvm.x86.avx2.maskload.d.256(ptr, <8 x i32>)


; The test is used to check MOV64rm instruction with relocation and ADD64rr_ND
; instruction are not folded to ADD64rm_ND with relocation. The later will emit
; APX relocation which is not recognized by the builtin linker on released OS.

; CHECK-LABEL: test_mem_fold
; NOAPXREL-NOT: R_X86_64_CODE_4_GOTPCRELX gvar3-0x4
; NOAPXREL: movq (%rip), %rbx
; NOAPXREL-NEXT: R_X86_64_REX_GOTPCRELX gvar3-0x4

@gvar3 = external global [40000 x i8]

define void @test_mem_fold(i32 %fetch.1644, i32 %sub.1142, i32 %mul.455, ptr %dval1, ptr %j1, ptr %j2, <4 x i1> %0, i1 %condloop.41.not, i32 %fetch.1646, i32 %fetch.1647, i32 %sub.1108, i64 %int_sext16, i64 %sub.1114, i1 %condloop.45.not.not, <4 x i1> %1) {
alloca_28:
  br label %ifmerge.52

do.body903:                                       ; preds = %ifmerge.2
  %mul.453 = mul i32 %sub.1108, %fetch.1647
  %sub.1144.neg = or i32 %mul.455, %fetch.1646
  %mul.454.neg = mul i32 %sub.1144.neg, %fetch.1644
  %sub.1147 = sub i32 0, %sub.1142
  %int_sext36 = sext i32 %mul.453 to i64
  %int_sext38 = sext i32 %mul.454.neg to i64
  %add.974 = or i64 %int_sext36, %int_sext38
  %div.98 = sdiv i64 %add.974, %int_sext16
  br label %do.body907

do.body907:                                       ; preds = %do.body907, %do.body903
  %do.count41.0 = phi i64 [ %sub.1173, %do.body907 ], [ %div.98, %do.body903 ]
  %gvar3.load = load double, ptr @gvar3, align 8
  store double %gvar3.load, ptr null, align 8
  call void (...) null(ptr null, ptr null, ptr null, ptr null, ptr %dval1, ptr null, ptr %j1, ptr %j2, ptr null, ptr null, ptr null, ptr null, ptr null, i64 0)
  store i32 %sub.1147, ptr null, align 4
  %sub.1173 = or i64 %do.count41.0, 1
  %rel.314 = icmp sgt i64 %do.count41.0, 0
  br label %do.body907

ifmerge.52:                                       ; preds = %ifmerge.2, %alloca_28
  %i1.i64.012 = phi i64 [ 0, %alloca_28 ], [ %sub.1114, %ifmerge.2 ]
  %2 = getelementptr double, ptr @gvar3, i64 %i1.i64.012
  br label %loop.45

loop.45:                                          ; preds = %loop.45, %ifmerge.52
  %3 = getelementptr double, ptr %2, <4 x i64> zeroinitializer
  %4 = call <4 x double> @llvm.masked.gather.v4f64.v4p0(<4 x ptr> %3, i32 0, <4 x i1> %0, <4 x double> zeroinitializer)
  call void @llvm.masked.scatter.v4f64.v4p0(<4 x double> %4, <4 x ptr> zeroinitializer, i32 0, <4 x i1> %0)
  br i1 %condloop.45.not.not, label %loop.45, label %ifmerge.2

ifmerge.2:                                        ; preds = %loop.45
  br i1 %condloop.41.not, label %do.body903, label %ifmerge.52
}

declare <4 x double> @llvm.masked.gather.v4f64.v4p0(<4 x ptr>, i32 immarg, <4 x i1>, <4 x double>)
declare void @llvm.masked.scatter.v4f64.v4p0(<4 x double>, <4 x ptr>, i32 immarg, <4 x i1>)


; The test is to check no R_X86_64_CODE_4_GOTPCRELX relocation emitted when the
; register in operand 0 of instruction with relocation is used in the PHI
; instruction. In PHI elimination pass, PHI instruction is eliminated by
; inserting COPY instruction. And in the late pass (Machine Copy Propagation
; pass), the COPY instruction may be optimized and the register in operand 0 of
; instruction with relocation may be replaced with EGPR.


; CHECK-LABEL: test_phi_uses
; APXREL: addq (%rip), %r16
; APXREL-NEXT: R_X86_64_CODE_4_GOTPCRELX gvar4-0x4
; APXREL: movq (%rip), %r17
; APXREL-NEXT: R_X86_64_CODE_4_GOTPCRELX gvar5-0x4
; APXREL: movq (%rip), %r18
; APXREL-NEXT: R_X86_64_CODE_4_GOTPCRELX gvar6-0x4
; APXREL: movq (%rip), %r19
; APXREL-NEXT: R_X86_64_CODE_4_GOTPCRELX gvar7-0x4
; APXREL: movq (%rip), %r22
; APXREL-NEXT: R_X86_64_CODE_4_GOTPCRELX gvar8-0x4
; APXREL: movq (%rip), %r23
; APXREL-NEXT: R_X86_64_CODE_4_GOTPCRELX gvar9-0x4
; APXREL: movq (%rip), %r24
; APXREL-NEXT: R_X86_64_CODE_4_GOTPCRELX gvar10-0x4
; NOAPXREL-NOT: R_X86_64_CODE_4_GOTPCRELX gvar5-0x4
; NOAPXREL: movq (%rip), %r15
; NOAPXREL-NEXT: R_X86_64_REX_GOTPCRELX gvar5-0x4


@gvar4 = external global [33 x [33 x double]]
@gvar5 = external global [33 x [33 x float]]
@gvar6 = external global [33 x [33 x float]]
@gvar7 = external global [33 x [33 x float]]
@gvar8 = external global [33 x [33 x float]]
@gvar9 = external global [33 x [33 x float]]
@gvar10 = external global [33 x [33 x float]]

define void @test_phi_uses(i64 %i1.i64.0, ptr %0, ptr %1, ptr %2, ptr %3, ptr %in0, ptr %4, ptr %5, i1 %cmp.144) #0 {
alloca_15:
  br label %loop.253

loop.253:                                         ; preds = %loop.1500, %alloca_15
  %i1.i64.01 = phi i64 [ 0, %alloca_15 ], [ %6, %loop.1500 ]
  %6 = add i64 %i1.i64.01, 1
  br label %loop.254

loop.254:                                         ; preds = %loop.254, %loop.253
  %i2.i64.02 = phi i64 [ %13, %loop.254 ], [ 0, %loop.253 ]
  %7 = getelementptr [33 x [33 x float]], ptr @gvar10, i64 0, i64 %i2.i64.02, i64 %i1.i64.01
  %gepload368 = load float, ptr %7, align 4
  store double 0.000000e+00, ptr %0, align 8
  %8 = getelementptr [33 x [33 x float]], ptr @gvar9, i64 0, i64 %i2.i64.02, i64 %i1.i64.01
  %gepload369 = load float, ptr %8, align 4
  store double 0.000000e+00, ptr %1, align 8
  %9 = getelementptr [33 x [33 x float]], ptr @gvar8, i64 0, i64 %i2.i64.02, i64 %i1.i64.01
  %gepload371 = load float, ptr %9, align 4
  store double 0.000000e+00, ptr %2, align 8
  %10 = getelementptr [33 x [33 x float]], ptr @gvar7, i64 0, i64 %i2.i64.02, i64 %i1.i64.01
  %gepload373 = load float, ptr %10, align 4
  %11 = getelementptr [33 x [33 x double]], ptr @gvar4, i64 0, i64 %i2.i64.02, i64 %i1.i64.0
  store double 0.000000e+00, ptr %11, align 8
  %12 = getelementptr [33 x [33 x float]], ptr @gvar6, i64 0, i64 %i2.i64.02, i64 %i1.i64.01
  %gepload375 = load float, ptr %12, align 4
  store double 0.000000e+00, ptr %3, align 8
  store double 0.000000e+00, ptr %5, align 8
  %13 = add i64 %i2.i64.02, 1
  store double 0.000000e+00, ptr %in0, align 8
  store double 0.000000e+00, ptr %4, align 8
  %14 = getelementptr [33 x [33 x float]], ptr @gvar5, i64 0, i64 %i2.i64.02, i64 %i1.i64.01
  %gepload392 = load float, ptr %14, align 4
  br i1 %cmp.144, label %loop.1500, label %loop.254

loop.1500:                                        ; preds = %loop.254
  %15 = getelementptr [33 x [33 x float]], ptr @gvar5, i64 0, i64 0, i64 %i1.i64.0
  %gepload444 = load float, ptr %15, align 4
  %16 = fpext float %gepload444 to double
  store double %16, ptr null, align 8
  br label %loop.253
}