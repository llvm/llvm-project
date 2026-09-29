; RUN: llc < %s -mattr=+ptx72 -O0 | FileCheck %s
;
;; Test same function inlined multiple times from different call sites - verifies that
;; inlined_at information correctly distinguishes multiple inline instances of rrand() and znew() called from different locations within a loop.
;
; typedef unsigned long long MYSIZE_T; // avoid platform dependence
; struct GridOpt {
;     int launch;
;     int sync;
;     int size;
;     MYSIZE_T itr;
; };
;
; enum LaunchOpt {
;     LaunchOptPerThreadStream = 0,
;     LaunchOptBlockSharedStream,
;     LaunchOptAllNullStream,
;     LaunchOptForkBomb,
;     LaunchOptLoopLaunch,
;     LaunchOptRand,
;     LaunchOptSize,
; };
;
; // post launch sync options
; enum SyncOpt {
;     SyncOptAllThreadsSync = 0,
;     SyncOptOneSyncPerBlock,
;     SyncOptRand,
;     SyncOptSize,
; };
;
; // size of launch options
; enum SizeOpt {
;     SizeOptSingleWarp = 0,
;     SizeOptMultiWarp,
;     SizeOptMultiBlock,
;     SizeOptRand,
;     SizeOptSize,
; };
;
;
; // device side failure codes
; enum ErrorStatus {
;     Success = 0,
;     DeviceRuntimeFailure  = 1,
;     DeviceMallocFailure   = 2,
;     DeviceHardwareFailure = 3,
;     InvalidInput          = 4,
; };
;
; __device__ MYSIZE_T znew( MYSIZE_T seed )
; {
;     return 36969 * ( seed & 65535 ) + seed >> 16;
; }
;
; __device__ MYSIZE_T rrand( MYSIZE_T *seed )
; {
;     *seed = znew( *seed );
;     return *seed;
; }
;
; __device__ GridOpt loopOpt;
;
; __global__ void cnpWideLaunch(GridOpt opt, MYSIZE_T maxLaunches, MYSIZE_T randomSeed, int *status, int *launchCounts)
; {
;     // MYSIZE_T threadSeed = randomSeed + blockDim.x * blockIdx.x + threadIdx.x;
;     MYSIZE_T blockSeed  = randomSeed + blockIdx.x;
;
;     // this device launch consumes a launch slot
;     maxLaunches--;
;
;     // compute number of launches per block
;     MYSIZE_T blockLaunches = maxLaunches  / gridDim.x;
;     MYSIZE_T extraLaunches = maxLaunches - gridDim.x * blockLaunches;
;     if (blockIdx.x < extraLaunches) {
;         blockLaunches++;
;     }
;
;     // clear launchcount with a thread in each block
;     if (threadIdx.x == 0)
;         launchCounts[blockIdx.x] = 0;
;
;     // compute per block random selections for sync/launch size/stream
;     for (MYSIZE_T i = 0; i < opt.itr; i++) {
;         loopOpt = opt;
;         if (opt.launch == LaunchOptRand) {
;             loopOpt.launch = rrand(&blockSeed) % (LaunchOptSize - 1);
;         }
;         if (opt.sync == SyncOptRand) {
;             loopOpt.sync = rrand(&blockSeed) % (SyncOptSize - 1);
;         }
;         if (opt.size == SizeOptRand) {
;             loopOpt.size = rrand(&blockSeed) % (SizeOptSize - 1);
;         }
;         __syncthreads();
;         if (threadIdx.x == 0) {
;             //printf("block %d launchCount %d blockLaunches:%d\n", blockIdx.x, launchCounts[blockIdx.x], blockLaunches);
;             int launchCount = launchCounts[blockIdx.x];
;
;             // fail if block did not generate enough launches
;             if (!*status && (launchCount != blockLaunches)) {
;                 *status = DeviceHardwareFailure;
;             }
;             // clear last iteration's launch
;             launchCounts[blockIdx.x] = 0;
;         }
;         __syncthreads();
;
;         // fail if a global error is present
;         if (*status) {
;             return;
;         }
;     }
; }
;
; CHECK: .loc [[FILENUM:[1-9]]] 84
; CHECK: .loc [[FILENUM]] 55 {{[0-9]*}}, function_name [[RRANDNAME:\$L__info_string[0-9]+]], inlined_at [[FILENUM]] 84
; CHECK: .loc [[FILENUM]] 50 {{[0-9]*}}, function_name [[ZNEWNAME:\$L__info_string[0-9]+]], inlined_at [[FILENUM]] 55
; CHECK: .loc [[FILENUM:[1-9]]] 87
; CHECK: .loc [[FILENUM]] 55 {{[0-9]*}}, function_name [[RRANDNAME]], inlined_at [[FILENUM]] 87
; CHECK: .loc [[FILENUM]] 50 {{[0-9]*}}, function_name [[ZNEWNAME]], inlined_at [[FILENUM]] 55
; CHECK: .loc [[FILENUM:[1-9]]] 90
; CHECK: .loc [[FILENUM]] 55 {{[0-9]*}}, function_name [[RRANDNAME]], inlined_at [[FILENUM]] 90
; CHECK: .loc [[FILENUM]] 50 {{[0-9]*}}, function_name [[ZNEWNAME]], inlined_at [[FILENUM]] 55
; CHECK: .section .debug_str
; CHECK: {
; CHECK: [[RRANDNAME]]:
; CHECK-NEXT: // {{.*}} _Z5rrandPy
; CHECK: [[ZNEWNAME]]:
; CHECK-NEXT: // {{.*}} _Z4znewy
; CHECK: }

source_filename = "<unnamed>"
target datalayout = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8-p6:32:32"
target triple = "nvptx64-nvidia-cuda"

%struct.GridOpt = type { i32, i32, i32, i64 }

@loopOpt = internal addrspace(1) global %struct.GridOpt zeroinitializer, align 8
@llvm.used = appending global [2 x ptr] [ptr @_Z13cnpWideLaunch7GridOptyyPiS0_, ptr addrspacecast (ptr addrspace(1) @loopOpt to ptr)], section "llvm.metadata"

define void @_Z13cnpWideLaunch7GridOptyyPiS0_(%struct.GridOpt noundef %opt, i64 noundef %maxLaunches, i64 noundef %randomSeed, ptr noundef captures(none) %status, ptr noundef captures(none) %launchCounts) !dbg !4 {
entry:
  %0 = addrspacecast ptr %status to ptr addrspace(1)
  %1 = addrspacecast ptr %launchCounts to ptr addrspace(1)
  %opt.fca.0.extract = extractvalue %struct.GridOpt %opt, 0
  %opt.fca.1.extract = extractvalue %struct.GridOpt %opt, 1
  %opt.fca.2.extract = extractvalue %struct.GridOpt %opt, 2
  %opt.fca.3.extract = extractvalue %struct.GridOpt %opt, 3
  %2 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !dbg !6
  %conv = zext nneg i32 %2 to i64, !dbg !6
  %add = add i64 %randomSeed, %conv, !dbg !6
  %dec = add i64 %maxLaunches, -1, !dbg !8
  %3 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x(), !dbg !9
  %conv6 = zext nneg i32 %3 to i64, !dbg !9
  %div = udiv i64 %dec, %conv6, !dbg !9
  %mul = mul i64 %div, %conv6, !dbg !10
  %sub = sub i64 %dec, %mul, !dbg !10
  %cmp = icmp ugt i64 %sub, %conv, !dbg !11
  %inc = zext i1 %cmp to i64, !dbg !11
  %spec.select = add i64 %div, %inc, !dbg !11
  %4 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !dbg !12
  %cmp20 = icmp eq i32 %4, 0, !dbg !12
  br i1 %cmp20, label %if.then22, label %if.end26, !dbg !12

if.then22:                                        ; preds = %entry
  %getElem = getelementptr inbounds nuw i32, ptr addrspace(1) %1, i64 %conv, !dbg !13
  store i32 0, ptr addrspace(1) %getElem, align 4, !dbg !13
  br label %if.end26, !dbg !13

if.end26:                                         ; preds = %if.then22, %entry
  %cmp3021.not = icmp eq i64 %opt.fca.3.extract, 0, !dbg !15
  br i1 %cmp3021.not, label %return, label %for.body.preheader, !dbg !15

for.body.preheader:                               ; preds = %if.end26
  %cmp34 = icmp eq i32 %opt.fca.0.extract, 5
  %cmp41 = icmp eq i32 %opt.fca.1.extract, 2
  %cmp50 = icmp eq i32 %opt.fca.2.extract, 3
  %getElem11 = getelementptr inbounds nuw i32, ptr addrspace(1) %1, i64 %conv
  br label %for.body, !dbg !16

for.body:                                         ; preds = %for.body.preheader, %if.end83
  %i.023 = phi i64 [ %inc90, %if.end83 ], [ 0, %for.body.preheader ]
  %blockSeed.022 = phi i64 [ %blockSeed.3, %if.end83 ], [ %add, %for.body.preheader ]
  %opt.elt = extractvalue %struct.GridOpt %opt, 0, !dbg !20
  %opt.elt25 = extractvalue %struct.GridOpt %opt, 1, !dbg !20
  %5 = insertelement <2 x i32> poison, i32 %opt.elt, i64 0, !dbg !20
  %6 = insertelement <2 x i32> %5, i32 %opt.elt25, i64 1, !dbg !20
  store <2 x i32> %6, ptr addrspace(1) @loopOpt, align 8, !dbg !20
  %opt.elt26 = extractvalue %struct.GridOpt %opt, 2, !dbg !20
  store i32 %opt.elt26, ptr addrspace(1) getelementptr inbounds nuw (i8, ptr addrspace(1) @loopOpt, i64 8), align 8, !dbg !20
  %opt.elt27 = extractvalue %struct.GridOpt %opt, 3, !dbg !20
  store i64 %opt.elt27, ptr addrspace(1) getelementptr inbounds nuw (i8, ptr addrspace(1) @loopOpt, i64 16), align 8, !dbg !20
  br i1 %cmp34, label %if.then36, label %if.end38, !dbg !16

if.then36:                                        ; preds = %for.body
  %and.i = and i64 %blockSeed.022, 65535, !dbg !21
  %mul.i = mul nuw nsw i64 %and.i, 36969, !dbg !21
  %add.i = add i64 %mul.i, %blockSeed.022, !dbg !21
  %shr.i = lshr i64 %add.i, 16, !dbg !21
  %rem = urem i64 %shr.i, 5, !dbg !29
  %conv37 = trunc nuw nsw i64 %rem to i32, !dbg !29
  store i32 %conv37, ptr addrspace(1) @loopOpt, align 8, !dbg !29
  br label %if.end38, !dbg !29

if.end38:                                         ; preds = %if.then36, %for.body
  %blockSeed.1 = phi i64 [ %shr.i, %if.then36 ], [ %blockSeed.022, %for.body ], !dbg !30
  br i1 %cmp41, label %if.then43, label %if.end47, !dbg !30

if.then43:                                        ; preds = %if.end38
  %and.i13 = and i64 %blockSeed.1, 65535, !dbg !31
  %mul.i14 = mul nuw nsw i64 %and.i13, 36969, !dbg !31
  %add.i15 = add i64 %mul.i14, %blockSeed.1, !dbg !31
  %shr.i16 = lshr i64 %add.i15, 16, !dbg !31
  %7 = trunc i64 %shr.i16 to i32, !dbg !35
  %conv46 = and i32 %7, 1, !dbg !35
  store i32 %conv46, ptr addrspace(1) getelementptr inbounds nuw (i8, ptr addrspace(1) @loopOpt, i64 4), align 4, !dbg !35
  br label %if.end47, !dbg !35

if.end47:                                         ; preds = %if.then43, %if.end38
  %blockSeed.2 = phi i64 [ %shr.i16, %if.then43 ], [ %blockSeed.1, %if.end38 ], !dbg !36
  br i1 %cmp50, label %if.then52, label %if.end56, !dbg !36

if.then52:                                        ; preds = %if.end47
  %and.i17 = and i64 %blockSeed.2, 65535, !dbg !37
  %mul.i18 = mul nuw nsw i64 %and.i17, 36969, !dbg !37
  %add.i19 = add i64 %mul.i18, %blockSeed.2, !dbg !37
  %shr.i20 = lshr i64 %add.i19, 16, !dbg !37
  %rem54 = urem i64 %shr.i20, 3, !dbg !41
  %conv55 = trunc nuw nsw i64 %rem54 to i32, !dbg !41
  store i32 %conv55, ptr addrspace(1) getelementptr inbounds nuw (i8, ptr addrspace(1) @loopOpt, i64 8), align 8, !dbg !41
  br label %if.end56, !dbg !41

if.end56:                                         ; preds = %if.then52, %if.end47
  %blockSeed.3 = phi i64 [ %shr.i20, %if.then52 ], [ %blockSeed.2, %if.end47 ], !dbg !42
  tail call void @llvm.nvvm.barrier0(), !dbg !42
  br i1 %cmp20, label %if.then61, label %if.end83, !dbg !43

if.then61:                                        ; preds = %if.end56
  %tmp67 = load i32, ptr addrspace(1) %getElem11, align 4, !dbg !44
  %tmp69 = load i32, ptr addrspace(1) %0, align 4, !dbg !46
  %tobool.not = icmp eq i32 %tmp69, 0, !dbg !46
  %conv71 = sext i32 %tmp67 to i64, !dbg !46
  %cmp73 = icmp ne i64 %spec.select, %conv71, !dbg !46
  %or.cond = select i1 %tobool.not, i1 %cmp73, i1 false, !dbg !46
  br i1 %or.cond, label %if.then75, label %if.end77, !dbg !46

if.then75:                                        ; preds = %if.then61
  store i32 3, ptr addrspace(1) %0, align 4, !dbg !47
  br label %if.end77, !dbg !47

if.end77:                                         ; preds = %if.then61, %if.then75
  store i32 0, ptr addrspace(1) %getElem11, align 4, !dbg !49
  br label %if.end83, !dbg !49

if.end83:                                         ; preds = %if.end77, %if.end56
  tail call void @llvm.nvvm.barrier0(), !dbg !50
  %tmp85 = load i32, ptr addrspace(1) %0, align 4, !dbg !51
  %tobool86.not = icmp eq i32 %tmp85, 0, !dbg !51
  %inc90 = add nuw i64 %i.023, 1
  %cmp30 = icmp ult i64 %inc90, %opt.fca.3.extract
  %or.cond24 = select i1 %tobool86.not, i1 %cmp30, i1 false, !dbg !51
  br i1 %or.cond24, label %for.body, label %return, !dbg !51, !llvm.loop !52

return:                                           ; preds = %if.end83, %if.end26
  ret void, !dbg !54
}

declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()

declare noundef range(i32 1, -2147483648) i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()

declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x()

declare void @llvm.nvvm.barrier0()

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: DebugDirectivesOnly)
!1 = !DIFile(filename: "t6.cu", directory: "")
!2 = !{}
!3 = !{i32 1, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "cnpWideLaunch", linkageName: "_Z13cnpWideLaunch7GridOptyyPiS0_", scope: !1, file: !1, line: 61, type: !5, scopeLine: 61, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DISubroutineType(types: !2)
!6 = !DILocation(line: 64, column: 5, scope: !7)
!7 = distinct !DILexicalBlock(scope: !4, file: !1, line: 62, column: 1)
!8 = !DILocation(line: 67, column: 5, scope: !7)
!9 = !DILocation(line: 70, column: 5, scope: !7)
!10 = !DILocation(line: 71, column: 5, scope: !7)
!11 = !DILocation(line: 72, column: 5, scope: !7)
!12 = !DILocation(line: 77, column: 5, scope: !7)
!13 = !DILocation(line: 78, column: 9, scope: !14)
!14 = distinct !DILexicalBlock(scope: !7, file: !1, line: 77, column: 5)
!15 = !DILocation(line: 81, column: 5, scope: !7)
!16 = !DILocation(line: 83, column: 9, scope: !17)
!17 = distinct !DILexicalBlock(scope: !18, file: !1, line: 81, column: 5)
!18 = distinct !DILexicalBlock(scope: !19, file: !1, line: 81, column: 5)
!19 = distinct !DILexicalBlock(scope: !7, file: !1, line: 81, column: 5)
!20 = !DILocation(line: 82, column: 9, scope: !17)
!21 = !DILocation(line: 50, column: 5, scope: !22, inlinedAt: !24)
!22 = distinct !DILexicalBlock(scope: !23, file: !1, line: 49, column: 1)
!23 = distinct !DISubprogram(name: "znew", linkageName: "_Z4znewy", scope: !1, file: !1, line: 48, type: !5, scopeLine: 48, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!24 = distinct !DILocation(line: 55, column: 5, scope: !25, inlinedAt: !27)
!25 = distinct !DILexicalBlock(scope: !26, file: !1, line: 54, column: 1)
!26 = distinct !DISubprogram(name: "rrand", linkageName: "_Z5rrandPy", scope: !1, file: !1, line: 53, type: !5, scopeLine: 53, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!27 = distinct !DILocation(line: 84, column: 13, scope: !28)
!28 = distinct !DILexicalBlock(scope: !17, file: !1, line: 83, column: 9)
!29 = !DILocation(line: 84, column: 13, scope: !28)
!30 = !DILocation(line: 86, column: 9, scope: !17)
!31 = !DILocation(line: 50, column: 5, scope: !22, inlinedAt: !32)
!32 = distinct !DILocation(line: 55, column: 5, scope: !25, inlinedAt: !33)
!33 = distinct !DILocation(line: 87, column: 13, scope: !34)
!34 = distinct !DILexicalBlock(scope: !17, file: !1, line: 86, column: 9)
!35 = !DILocation(line: 87, column: 13, scope: !34)
!36 = !DILocation(line: 89, column: 9, scope: !17)
!37 = !DILocation(line: 50, column: 5, scope: !22, inlinedAt: !38)
!38 = distinct !DILocation(line: 55, column: 5, scope: !25, inlinedAt: !39)
!39 = distinct !DILocation(line: 90, column: 13, scope: !40)
!40 = distinct !DILexicalBlock(scope: !17, file: !1, line: 89, column: 9)
!41 = !DILocation(line: 90, column: 13, scope: !40)
!42 = !DILocation(line: 92, column: 9, scope: !17)
!43 = !DILocation(line: 93, column: 9, scope: !17)
!44 = !DILocation(line: 95, column: 13, scope: !45)
!45 = distinct !DILexicalBlock(scope: !17, file: !1, line: 93, column: 9)
!46 = !DILocation(line: 98, column: 13, scope: !45)
!47 = !DILocation(line: 99, column: 17, scope: !48)
!48 = distinct !DILexicalBlock(scope: !45, file: !1, line: 98, column: 13)
!49 = !DILocation(line: 102, column: 13, scope: !45)
!50 = !DILocation(line: 104, column: 9, scope: !17)
!51 = !DILocation(line: 107, column: 9, scope: !17)
!52 = distinct !{!52, !53}
!53 = !{!"llvm.loop.mustprogress"}
!54 = !DILocation(line: 111, column: 1, scope: !7)
