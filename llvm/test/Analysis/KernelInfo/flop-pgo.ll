; Check info on floating point operations.

; RUN: opt -pass-remarks=kernel-info -passes=kernel-info \
; RUN:     -disable-output %s 2>&1 | \
; RUN:   FileCheck -match-full-lines -implicit-check-not='floating point' %s

target datalayout = "e-i65:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Check function with neither profile data nor floating point operations.
define double @noFlopOrProf() !dbg !100 {
  ret double 0.000000e+00, !dbg !105
}
; CHECK: remark: test.c:1:0: in function 'noFlopOrProf', FloatingPointOpProfileCount = 0

; Check function with profile data but no floating point operations.
define double @noFlop() !dbg !200 !prof !202 {
  ret double 0.000000e+00, !dbg !205
}
; CHECK: remark: test.c:2:0: in function 'noFlop', FloatingPointOpProfileCount = 0

; Check function with floating point operations but no profile data.
define double @noProf() !dbg !300 {
  ; CHECK: remark: test.c:3:9: in function 'noProf', double 'fadd' ('%fadd') has no profile data
  %fadd = fadd double 0.000000e+00, 0.000000e+00, !dbg !304
  ret double 0.000000e+00, !dbg !305
}
; CHECK: remark: test.c:3:0: in function 'noProf', FloatingPointOpProfileCount = 0

; Check function with floating point operations and profile data.
define double @f() !dbg !400 !prof !402 {
  ; Check floating point operation in entry block, which has a count of 1 per
  ; entry into the function.
  ;
  ; Also, check case of basic block with exactly 1 floating point operation.
  %alloca = alloca double, align 8, addrspace(1), !dbg !498
  ; CHECK: remark: test.c:5:9: in function 'f', double 'fadd' ('%fadd') executed 2 times
  %fadd = fadd double 0.000000e+00, 0.000000e+00, !dbg !405
  br label %.none, !dbg !498

  ; Check floating point operation in ret block.
  ;
  ; branch_weights gives this block a count of 1 per entry into the function.
.ret:                                         ; preds = %.many
  ; CHECK: remark: test.c:6:9: in function 'f', double 'fsub' ('%fsub') executed 2 times
  %fsub = fsub double 0.000000e+00, 0.000000e+00, !dbg !406
  ; CHECK: remark: test.c:7:9: in function 'f', double 'fmul' ('%fmul') executed 2 times
  %fmul = fmul double 0.000000e+00, 0.000000e+00, !dbg !407
  ret double 0.000000e+00, !dbg !498

  ; Check case of 0 floating point operations in a basic block.
.none:                                         ; preds = %0
  br label %.many, !dbg !498

  ; Check case of many floating point operations in a basic block.
  ;
  ; branch_weights gives this block a count of 3 per entry into the function.
.many:                                         ; preds = %.none, %.many
  ; These are not considered floating point ops even though they return floating
  ; point values.
  %phi = phi double [ %fadd, %.none ], [ %load, %.many ], !dbg !498
  %load = load double, ptr addrspace(1) %alloca, align 8, !dbg !498

  ; Check simple floating point ops not already checked above, and check an
  ; unnamed value.
  ;
  ; CHECK: remark: test.c:8:9: in function 'f', double 'fdiv' ('%1') executed 6 times
  %1 = fdiv double 0.000000e+00, 0.000000e+00, !dbg !408
  ; CHECK: remark: test.c:9:9: in function 'f', double 'fneg' ('%fneg') executed 6 times
  %fneg = fneg double 0.000000e+00, !dbg !409

  ; Check atomicrmw.
  ;
  ; CHECK: remark: test.c:10:9: in function 'f', double 'atomicrmw' ('%[[#]]') executed 6 times
  atomicrmw fadd ptr addrspace(37) null, double 0.000000e+00 seq_cst, !dbg !410
  ; CHECK: remark: test.c:11:9: in function 'f', double 'atomicrmw' ('%[[#]]') executed 6 times
  atomicrmw fsub ptr addrspace(37) null, double 0.000000e+00 seq_cst, !dbg !411
  ; CHECK: remark: test.c:12:9: in function 'f', double 'atomicrmw' ('%[[#]]') executed 6 times
  atomicrmw fmax ptr addrspace(37) null, double 0.000000e+00 seq_cst, !dbg !412
  ; CHECK: remark: test.c:13:9: in function 'f', double 'atomicrmw' ('%[[#]]') executed 6 times
  atomicrmw fmin ptr addrspace(37) null, double 0.000000e+00 seq_cst, !dbg !413
  ; atomicrmw that is not a floating point op.
  atomicrmw add ptr addrspace(37) null, i32 10 seq_cst, !dbg !498

  ; Check some intrinsics.
  ;
  ; CHECK: remark: test.c:14:9: in function 'f', double 'llvm.sqrt.f64' call ('%sqrt') executed 6 times
  %sqrt = call double @llvm.sqrt.f64(double 0.000000e+00), !dbg !414
  ; CHECK: remark: test.c:15:9: in function 'f', double 'llvm.sin.f64' call ('%sin') executed 6 times
  %sin = call double @llvm.sin.f64(double 0.000000e+00), !dbg !415
  ; CHECK: remark: test.c:16:9: in function 'f', double 'llvm.fmuladd.f64' call ('%fmuladd') executed 6 times
  %fmuladd = call double @llvm.fmuladd.f64(double 0.000000e+00, double 0.000000e+00, double 0.000000e+00), !dbg !416
  ; Intrinsic that is not a floating point op.
  %umax = call i32 @llvm.umax.i32(i32 0, i32 0), !dbg !498

  ; Check floating point types besides double scalar.
  ;
  ; CHECK: remark: test.c:50:9: in function 'f', float 'fadd' ('%float') executed 6 times
  %float = fadd float 0.000000e+00, 0.000000e+00, !dbg !450
  ; CHECK: remark: test.c:51:9: in function 'f', half 'fadd' ('%half') executed 6 times
  %half = fadd half 0.000000e+00, 0.000000e+00, !dbg !451
  ; CHECK: remark: test.c:52:9: in function 'f', bfloat 'fadd' ('%bfloat') executed 6 times
  %bfloat = fadd bfloat 0.000000e+00, 0.000000e+00, !dbg !452
  ; CHECK: remark: test.c:53:9: in function 'f', fp128 'fadd' ('%fp128') executed 6 times
  %fp128 = fadd fp128 0xL0, 0xL0, !dbg !453
  ; CHECK: remark: test.c:54:9: in function 'f', <2 x double> 'fadd' ('%vector') executed 6 times
  %vector = fadd <2 x double> <double 0.000000e+00, double 0.000000e+00>, <double 0.000000e+00, double 0.000000e+00>, !dbg !454

  br i1 false, label %.ret, label %.many, !prof !499, !dbg !498
}
; CHECK: remark: test.c:4:0: in function 'f', FloatingPointOpProfileCount = 90

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C11, file: !2, producer: "clang version 20.0.0git", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!2 = !DIFile(filename: "test.c", directory: "/tmp")
!3 = !{}

!100 = distinct !DISubprogram(name: "noFlopOrProf", scope: !2, file: !2, line: 1, type: !101, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1, retainedNodes: !3)
!101 = !DISubroutineType(types: !3)
!103 = distinct !DILexicalBlock(scope: !104, file: !2, line: 1, column: 3)
!104 = distinct !DILexicalBlock(scope: !100, file: !2, line: 1, column: 3)
!105 = !DILocation(line: 1, column: 9, scope: !103)

!200 = distinct !DISubprogram(name: "noFlop", scope: !2, file: !2, line: 2, type: !201, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1, retainedNodes: !3)
!201 = !DISubroutineType(types: !3)
!202 = !{!"function_entry_count", i64 5}
!203 = distinct !DILexicalBlock(scope: !204, file: !2, line: 2, column: 3)
!204 = distinct !DILexicalBlock(scope: !200, file: !2, line: 2, column: 3)
!205 = !DILocation(line: 2, column: 9, scope: !203)

!300 = distinct !DISubprogram(name: "noProf", scope: !2, file: !2, line: 3, type: !301, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1, retainedNodes: !3)
!301 = !DISubroutineType(types: !3)
!302 = distinct !DILexicalBlock(scope: !303, file: !2, line: 3, column: 3)
!303 = distinct !DILexicalBlock(scope: !300, file: !2, line: 3, column: 3)
!304 = !DILocation(line: 3, column: 9, scope: !302)
!305 = !DILocation(line: 4, column: 9, scope: !302)

!400 = distinct !DISubprogram(name: "f", scope: !2, file: !2, line: 4, type: !401, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1, retainedNodes: !3)
!401 = !DISubroutineType(types: !3)
!402 = !{!"function_entry_count", i64 2}
!403 = distinct !DILexicalBlock(scope: !404, file: !2, line: 6, column: 3)
!404 = distinct !DILexicalBlock(scope: !400, file: !2, line: 6, column: 3)
!405 = !DILocation(line: 5, column: 9, scope: !403)
!406 = !DILocation(line: 6, column: 9, scope: !403)
!407 = !DILocation(line: 7, column: 9, scope: !403)
!408 = !DILocation(line: 8, column: 9, scope: !403)
!409 = !DILocation(line: 9, column: 9, scope: !403)
!410 = !DILocation(line: 10, column: 9, scope: !403)
!411 = !DILocation(line: 11, column: 9, scope: !403)
!412 = !DILocation(line: 12, column: 9, scope: !403)
!413 = !DILocation(line: 13, column: 9, scope: !403)
!414 = !DILocation(line: 14, column: 9, scope: !403)
!415 = !DILocation(line: 15, column: 9, scope: !403)
!416 = !DILocation(line: 16, column: 9, scope: !403)
!450 = !DILocation(line: 50, column: 9, scope: !403)
!451 = !DILocation(line: 51, column: 9, scope: !403)
!452 = !DILocation(line: 52, column: 9, scope: !403)
!453 = !DILocation(line: 53, column: 9, scope: !403)
!454 = !DILocation(line: 54, column: 9, scope: !403)
!498 = !DILocation(line: 999, column: 999, scope: !403)
!499 = !{!"branch_weights", i32 127, i32 257}
