; Check info on floating point operations.

; RUN: opt -pass-remarks=kernel-info -passes=kernel-info \
; RUN:     -disable-output %s 2>&1 | \
; RUN:   FileCheck %s -match-full-lines \
; RUN:     -implicit-check-not='executed {{[0-9]+}} flops' \
; RUN:     -implicit-check-not='moved {{[0-9]+}} fp bytes'

target datalayout = "e-i65:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Check function with neither profile data nor floating point operations.
define double @noFlopOrProf() !dbg !100 {
  ret double 0.000000e+00, !dbg !105
}
; CHECK: remark: test.c:1:0: in function 'noFlopOrProf', ProfileFloatingPointOpCount = 0
; CHECK: remark: test.c:1:0: in function 'noFlopOrProf', ProfileFloatingPointBytesMoved = 0

; Check function with profile data but no floating point operations.
define double @noFlop() !dbg !200 !prof !202 {
  ret double 0.000000e+00, !dbg !205
}
; CHECK: remark: test.c:2:0: in function 'noFlop', ProfileFloatingPointOpCount = 0
; CHECK: remark: test.c:2:0: in function 'noFlop', ProfileFloatingPointBytesMoved = 0

; Check function with floating point operations (one that moves data and one
; that does not) but no profile data.
define double @noProf() !dbg !300 {
  ; CHECK: remark: test.c:4:9: in function 'noProf', double 'fadd' ('%fadd') has no profile data
  %fadd = fadd double 0.000000e+00, 0.000000e+00, !dbg !304
  ; CHECK: remark: test.c:5:9: in function 'noProf', double 'load' ('%load') has no profile data
  %load = load double, ptr null, align 4, !dbg !305
  ret double 0.000000e+00, !dbg !306
}
; CHECK: remark: test.c:3:0: in function 'noProf', ProfileFloatingPointOpCount = 0
; CHECK: remark: test.c:3:0: in function 'noProf', ProfileFloatingPointBytesMoved = 0

; Check function with floating point operations and profile data.
define double @f() !dbg !400 !prof !402 {
  ; Check floating point operation in entry block, which has a count of 1 per
  ; entry into the function.
  ;
  ; Also, check case of basic block with exactly 1 floating point operation.
  %alloca = alloca double, align 8, addrspace(1), !dbg !405
  ; CHECK: remark: test.c:10:9: in function 'f', double 'fadd' ('%fadd') executed 2 flops
  %fadd = fadd double 0.000000e+00, 0.000000e+00, !dbg !410
  br label %.none, !dbg !405

  ; Check floating point operation in ret block.
  ;
  ; branch_weights gives this block a count of 1 per entry into the function.
.ret:                                         ; preds = %.many
  ; CHECK: remark: test.c:20:9: in function 'f', double 'fsub' ('%fsub') executed 2 flops
  %fsub = fsub double 0.000000e+00, 0.000000e+00, !dbg !420
  ; CHECK: remark: test.c:21:9: in function 'f', double 'fmul' ('%fmul') executed 2 flops
  %fmul = fmul double 0.000000e+00, 0.000000e+00, !dbg !421
  ret double 0.000000e+00, !dbg !405

  ; Check case of 0 floating point operations in a basic block.
.none:                                         ; preds = %0
  br label %.many, !dbg !405

  ; Check case of many floating point operations in a basic block.
  ;
  ; branch_weights gives this block a count of 3 per entry into the function.
.many:                                         ; preds = %.none, %.many
  ; These are not counted as floating point ops even though they return floating
  ; point values.  For AMD GPUs, we have seen no evidence that the hardware
  ; instructions to which they lower ever trigger floating point hardware
  ; counters.  More appear with conversions below.
  %phi = phi double [ %fadd, %.none ], [ %load, %.many ], !dbg !405
  %fneg = fneg double 0.000000e+00, !dbg !405
  %copysign = call double @llvm.copysign.f64(double 0.000000e+00, double 0.000000e+00), !dbg !405
  %fabs = call double @llvm.fabs.f64(double 0.000000e+00), !dbg !405
  %floor = call double @llvm.floor.f64(double 0.000000e+00), !dbg !405
  %ldexp = call double @llvm.ldexp.f64.i32(double 0.000000e+00, i32 0), !dbg !405
  %minnum = call double @llvm.minnum.f64(double 0.000000e+00, double 0.000000e+00), !dbg !405
  %rint = call double @llvm.rint.f64(double 0.000000e+00), !dbg !405

  ; Check simple floating point ops not already checked above, and check an
  ; unnamed value.
  ;
  ; CHECK: remark: test.c:30:9: in function 'f', double 'fdiv' ('%1') executed 6 flops x 14
  %1 = fdiv double 0.000000e+00, 0.000000e+00, !dbg !430
  ; CHECK: remark: test.c:31:9: in function 'f', double 'load' ('%load') moved 48 fp bytes
  %load = load double, ptr addrspace(1) %alloca, align 8, !dbg !431
  ; CHECK: remark: test.c:32:9: in function 'f', double 'store' moved 48 fp bytes
  store double 0.000000e+00, ptr addrspace(1) %alloca, align 8, !dbg !432

  ; Check atomicrmw.
  ;
  ; CHECK: remark: test.c:40:9: in function 'f', double 'atomicrmw' ('%[[#]]') moved 48 fp bytes
  atomicrmw xchg ptr addrspace(37) null, double 0.000000e+00 seq_cst, !dbg !440
  ; CHECK: remark: test.c:41:9: in function 'f', double 'atomicrmw' ('%[[#]]') moved 48 fp bytes
  ; CHECK: remark: test.c:41:9: in function 'f', double 'atomicrmw' ('%[[#]]') executed 6 flops
  atomicrmw fadd ptr addrspace(37) null, double 0.000000e+00 seq_cst, !dbg !441
  ; CHECK: remark: test.c:42:9: in function 'f', double 'atomicrmw' ('%[[#]]') moved 48 fp bytes
  ; CHECK: remark: test.c:42:9: in function 'f', double 'atomicrmw' ('%[[#]]') executed 6 flops
  atomicrmw fsub ptr addrspace(37) null, double 0.000000e+00 seq_cst, !dbg !442
  ; CHECK: remark: test.c:43:9: in function 'f', double 'atomicrmw' ('%[[#]]') moved 48 fp bytes
  ; CHECK: remark: test.c:43:9: in function 'f', double 'atomicrmw' ('%[[#]]') executed 6 flops
  atomicrmw fmax ptr addrspace(37) null, double 0.000000e+00 seq_cst, !dbg !443
  ; CHECK: remark: test.c:44:9: in function 'f', double 'atomicrmw' ('%[[#]]') moved 48 fp bytes
  ; CHECK: remark: test.c:44:9: in function 'f', double 'atomicrmw' ('%[[#]]') executed 6 flops
  atomicrmw fmin ptr addrspace(37) null, double 0.000000e+00 seq_cst, !dbg !444
  ; atomicrmw that is not a floating point op.
  atomicrmw add ptr addrspace(37) null, i32 10 seq_cst, !dbg !405

  ; Check some flop intrinsics.
  ;
  ; CHECK: remark: test.c:50:9: in function 'f', double 'llvm.amdgcn.rcp.f64' call ('%rcp') executed 6 flops
  %rcp = call double @llvm.amdgcn.rcp.f64(double 0.000000e+00), !dbg !450
  ; CHECK: remark: test.c:51:9: in function 'f', double 'llvm.amdgcn.trig.preop.f64' call ('%trig.preop') executed 6 flops
  %trig.preop = call double @llvm.amdgcn.trig.preop.f64(double 0.000000e+00, i32 0), !dbg !451
  ; CHECK: remark: test.c:52:9: in function 'f', double 'llvm.fma.f64' call ('%fma') executed 6 flops x 2
  %fma = call double @llvm.fma.f64(double 0.000000e+00, double 0.000000e+00, double 0.000000e+00), !dbg !452
  ; CHECK: remark: test.c:53:9: in function 'f', double 'llvm.fmuladd.f64' call ('%fmuladd') executed 6 flops x 2
  %fmuladd = call double @llvm.fmuladd.f64(double 0.000000e+00, double 0.000000e+00, double 0.000000e+00), !dbg !453
  ; CHECK: remark: test.c:54:9: in function 'f', double 'llvm.sin.f64' call ('%sin') executed 6 flops
  %sin = call double @llvm.sin.f64(double 0.000000e+00), !dbg !454
  ; CHECK: remark: test.c:55:9: in function 'f', double 'llvm.sqrt.f64' call ('%sqrt') executed 6 flops x 17
  %sqrt = call double @llvm.sqrt.f64(double 0.000000e+00), !dbg !455
  ; Intrinsic that is not a floating point op.
  %umax = call i32 @llvm.umax.i32(i32 0, i32 0), !dbg !405

  ; Check floating point types besides double scalar.
  ;
  ; CHECK: remark: test.c:60:9: in function 'f', float 'fadd' ('%float') executed 6 flops
  %float = fadd float 0.000000e+00, 0.000000e+00, !dbg !460
  ; CHECK: remark: test.c:61:9: in function 'f', float 'store' moved 24 fp bytes
  store float 0.000000e+00, ptr null, align 8, !dbg !461
  ; CHECK: remark: test.c:62:9: in function 'f', half 'fadd' ('%half') executed 6 flops
  %half = fadd half 0.000000e+00, 0.000000e+00, !dbg !462
  ; CHECK: remark: test.c:63:9: in function 'f', half 'store' moved 12 fp bytes
  store half 0.000000e+00, ptr null, align 8, !dbg !463
  ; CHECK: remark: test.c:64:9: in function 'f', bfloat 'fadd' ('%bfloat') executed 6 flops
  %bfloat = fadd bfloat 0.000000e+00, 0.000000e+00, !dbg !464
  ; CHECK: remark: test.c:65:9: in function 'f', bfloat 'store' moved 12 fp bytes
  store bfloat 0.000000e+00, ptr null, align 8, !dbg !465
  ; CHECK: remark: test.c:66:9: in function 'f', fp128 'fadd' ('%fp128') executed 6 flops
  %fp128 = fadd fp128 0xL0, 0xL0, !dbg !466
  ; CHECK: remark: test.c:67:9: in function 'f', fp128 'store' moved 96 fp bytes
  store fp128 0xL0, ptr null, align 8, !dbg !467
  ; CHECK: remark: test.c:68:9: in function 'f', <2 x double> 'fadd' ('%vector') executed 6 flops
  %vector = fadd <2 x double> <double 0.000000e+00, double 0.000000e+00>, <double 0.000000e+00, double 0.000000e+00>, !dbg !468
  ; CHECK: remark: test.c:69:9: in function 'f', <2 x double> 'store' moved 96 fp bytes
  store <2 x double> <double 0.000000e+00, double 0.000000e+00>, ptr null, align 8, !dbg !469

  ; Check conversions.
  ;
  ; CHECK: remark: test.c:70:9: in function 'f', double 'uitofp' ('%uitofp.64.64') executed 6 flops
  %uitofp.64.64 = uitofp i64 0 to double, !dbg !470
  ; CHECK: remark: test.c:71:9: in function 'f', double 'sitofp' ('%sitofp.64.64') executed 6 flops
  %sitofp.64.64 = sitofp i64 0 to double, !dbg !471
  ; CHECK: remark: test.c:72:9: in function 'f', double 'fptoui' ('%fptoui.64.64') executed 6 flops x 2
  %fptoui.64.64 = fptoui double 0.000000e+00 to i64, !dbg !472
  ; CHECK: remark: test.c:73:9: in function 'f', double 'fptosi' ('%fptosi.64.64') executed 6 flops x 2
  %fptosi.64.64 = fptosi double 0.000000e+00 to i64, !dbg !473
  %uitofp.32.64 = uitofp i32 0 to double, !dbg !405
  %sitofp.32.64 = sitofp i32 0 to double, !dbg !405
  %fptoui.64.32 = fptoui double 0.000000e+00 to i32, !dbg !405
  %fptosi.64.32 = fptosi double 0.000000e+00 to i32, !dbg !405
  %uitofp.64.32 = uitofp i64 0 to float, !dbg !405
  %sitofp.64.32 = sitofp i64 0 to float, !dbg !405
  %fptoui.32.64 = fptoui float 0.000000e+00 to i64, !dbg !405
  %fptosi.32.64 = fptosi float 0.000000e+00 to i64, !dbg !405
  %uitofp.32.32 = uitofp i32 0 to float, !dbg !405
  %sitofp.32.32 = sitofp i32 0 to float, !dbg !405
  %fptoui.32.32 = fptoui float 0.000000e+00 to i32, !dbg !405
  %fptosi.32.32 = fptosi float 0.000000e+00 to i32, !dbg !405
  %fptrunc.64.32 = fptrunc double 0.000000e+00 to float, !dbg !405
  %fpext.32.64 = fpext float 0.000000e+00 to double, !dbg !405
  %bitcast.double.i64 = bitcast double 0.000000e+00 to i64, !dbg !405
  %bitcast.i64.double = bitcast i64 0 to double, !dbg !405

  br i1 false, label %.ret, label %.many, !prof !499, !dbg !405
}
; CHECK: remark: test.c:4:0: in function 'f', ProfileFloatingPointOpCount = 324
; CHECK: remark: test.c:4:0: in function 'f', ProfileFloatingPointBytesMoved = 576

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
!304 = !DILocation(line: 4, column: 9, scope: !302)
!305 = !DILocation(line: 5, column: 9, scope: !302)
!306 = !DILocation(line: 6, column: 9, scope: !302)

!400 = distinct !DISubprogram(name: "f", scope: !2, file: !2, line: 4, type: !401, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1, retainedNodes: !3)
!401 = !DISubroutineType(types: !3)
!402 = !{!"function_entry_count", i64 2}
!403 = distinct !DILexicalBlock(scope: !404, file: !2, line: 6, column: 3)
!404 = distinct !DILexicalBlock(scope: !400, file: !2, line: 6, column: 3)
!405 = !DILocation(line: 999, column: 999, scope: !403)
!410 = !DILocation(line: 10, column: 9, scope: !403)
!420 = !DILocation(line: 20, column: 9, scope: !403)
!421 = !DILocation(line: 21, column: 9, scope: !403)
!430 = !DILocation(line: 30, column: 9, scope: !403)
!431 = !DILocation(line: 31, column: 9, scope: !403)
!432 = !DILocation(line: 32, column: 9, scope: !403)
!440 = !DILocation(line: 40, column: 9, scope: !403)
!441 = !DILocation(line: 41, column: 9, scope: !403)
!442 = !DILocation(line: 42, column: 9, scope: !403)
!443 = !DILocation(line: 43, column: 9, scope: !403)
!444 = !DILocation(line: 44, column: 9, scope: !403)
!450 = !DILocation(line: 50, column: 9, scope: !403)
!451 = !DILocation(line: 51, column: 9, scope: !403)
!452 = !DILocation(line: 52, column: 9, scope: !403)
!453 = !DILocation(line: 53, column: 9, scope: !403)
!454 = !DILocation(line: 54, column: 9, scope: !403)
!455 = !DILocation(line: 55, column: 9, scope: !403)
!460 = !DILocation(line: 60, column: 9, scope: !403)
!461 = !DILocation(line: 61, column: 9, scope: !403)
!462 = !DILocation(line: 62, column: 9, scope: !403)
!463 = !DILocation(line: 63, column: 9, scope: !403)
!464 = !DILocation(line: 64, column: 9, scope: !403)
!465 = !DILocation(line: 65, column: 9, scope: !403)
!466 = !DILocation(line: 66, column: 9, scope: !403)
!467 = !DILocation(line: 67, column: 9, scope: !403)
!468 = !DILocation(line: 68, column: 9, scope: !403)
!469 = !DILocation(line: 69, column: 9, scope: !403)
!470 = !DILocation(line: 70, column: 9, scope: !403)
!471 = !DILocation(line: 71, column: 9, scope: !403)
!472 = !DILocation(line: 72, column: 9, scope: !403)
!473 = !DILocation(line: 73, column: 9, scope: !403)
!499 = !{!"branch_weights", i32 1, i32 2}
