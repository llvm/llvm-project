; RUN: llc -verify-machineinstrs < %s | FileCheck %s
; This test case used to fail both with and without -verify-machineinstrs
; (-verify-machineinstrs would catch the problem right after instruction
; scheduling because the live intervals would not be right for the registers
; that were both inputs to the inline asm and also early-clobber outputs).

target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-bgq-linux"

%struct._IO_FILE.119.8249.32639.195239.200117.211499.218003.221255.222881.224507.226133.240767.244019.245645.248897.260279.271661.281417.283043.302555.304181.325319.326945.344713 = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker.118.8248.32638.195238.200116.211498.218002.221254.222880.224506.226132.240766.244018.245644.248896.260278.271660.281416.283042.302554.304180.325318.326944.344712*, %struct._IO_FILE.119.8249.32639.195239.200117.211499.218003.221255.222881.224507.226133.240767.244019.245645.248897.260279.271661.281417.283043.302555.304181.325319.326945.344713*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker.118.8248.32638.195238.200116.211498.218002.221254.222880.224506.226132.240766.244018.245644.248896.260278.271660.281416.283042.302554.304180.325318.326944.344712 = type { %struct._IO_marker.118.8248.32638.195238.200116.211498.218002.221254.222880.224506.226132.240766.244018.245644.248896.260278.271660.281416.283042.302554.304180.325318.326944.344712*, %struct._IO_FILE.119.8249.32639.195239.200117.211499.218003.221255.222881.224507.226133.240767.244019.245645.248897.260279.271661.281417.283043.302555.304181.325319.326945.344713*, i32 }

@.str236 = external unnamed_addr constant [121 x i8], align 1
@.str294 = external unnamed_addr constant [49 x i8], align 1

; Function Attrs: nounwind
declare void @fprintf(%struct._IO_FILE.119.8249.32639.195239.200117.211499.218003.221255.222881.224507.226133.240767.244019.245645.248897.260279.271661.281417.283043.302555.304181.325319.326945.344713* nocapture, i8* nocapture readonly, ...) #0

; Function Attrs: inlinehint nounwind
define void @_ZN4PAMI6Device2MU15ResourceManager46calculatePerCoreMUResourcesBasedOnAvailabilityEv(i32 %inp32, i64 %inp64) #1 align 2 {
; CHECK-LABEL: @_ZN4PAMI6Device2MU15ResourceManager46calculatePerCoreMUResourcesBasedOnAvailabilityEv
; CHECK: sc

entry:
  %numFreeResourcesInSubgroup = alloca i32, align 4
  %0 = ptrtoint i32* %numFreeResourcesInSubgroup to i64
  br label %for.cond2.preheader

for.cond2.preheader:                              ; preds = %if.end23.3, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %if.end23.3 ]
  %group.098 = phi i32 [ 0, %entry ], [ %inc37, %if.end23.3 ]
  %minFreeBatIdsPerCore.097 = phi i64 [ 32, %entry ], [ %numFreeBatIdsInGroup.0.minFreeBatIdsPerCore.0, %if.end23.3 ]
  %minFreeRecFifosPerCore.096 = phi i64 [ 16, %entry ], [ %minFreeRecFifosPerCore.1, %if.end23.3 ]
  %minFreeInjFifosPerCore.095 = phi i64 [ 32, %entry ], [ %numFreeInjFifosInGroup.0.minFreeInjFifosPerCore.0, %if.end23.3 ]
  %cmp5 = icmp eq i32 %inp32, 0
  br i1 %cmp5, label %if.end, label %if.then

if.then:                                          ; preds = %if.end23.2, %if.end23.1, %if.end23, %for.cond2.preheader
  unreachable

if.end:                                           ; preds = %for.cond2.preheader
  %1 = load i32, i32* %numFreeResourcesInSubgroup, align 4
  %conv = zext i32 %1 to i64
  %2 = call { i64, i64, i64, i64 } asm sideeffect "sc", "=&{r0},=&{r3},=&{r4},=&{r5},{r0},{r3},{r4},{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{r12},~{cr0},~{memory}"(i64 1034, i64 %indvars.iv, i64 %0, i64 %inp64) #2
  %cmp10 = icmp eq i32 0, 0
  br i1 %cmp10, label %if.end14, label %if.then11

if.then11:                                        ; preds = %if.end.3, %if.end.2, %if.end.1, %if.end
  unreachable

if.end14:                                         ; preds = %if.end
  %3 = load i32, i32* %numFreeResourcesInSubgroup, align 4
  %cmp19 = icmp eq i32 %inp32, 0
  br i1 %cmp19, label %if.end23, label %if.then20

if.then20:                                        ; preds = %if.end14.3, %if.end14.2, %if.end14.1, %if.end14
  %conv4.i65.lcssa = phi i32 [ %inp32, %if.end14 ], [ 0, %if.end14.1 ], [ %conv4.i65.2, %if.end14.2 ], [ %conv4.i65.3, %if.end14.3 ]
  call void (%struct._IO_FILE.119.8249.32639.195239.200117.211499.218003.221255.222881.224507.226133.240767.244019.245645.248897.260279.271661.281417.283043.302555.304181.325319.326945.344713*, i8*, ...) @fprintf(%struct._IO_FILE.119.8249.32639.195239.200117.211499.218003.221255.222881.224507.226133.240767.244019.245645.248897.260279.271661.281417.283043.302555.304181.325319.326945.344713* undef, i8* getelementptr inbounds ([121 x i8], [121 x i8]* @.str236, i64 0, i64 0), i32 signext 2503) #3
  call void (%struct._IO_FILE.119.8249.32639.195239.200117.211499.218003.221255.222881.224507.226133.240767.244019.245645.248897.260279.271661.281417.283043.302555.304181.325319.326945.344713*, i8*, ...) @fprintf(%struct._IO_FILE.119.8249.32639.195239.200117.211499.218003.221255.222881.224507.226133.240767.244019.245645.248897.260279.271661.281417.283043.302555.304181.325319.326945.344713* undef, i8* getelementptr inbounds ([49 x i8], [49 x i8]* @.str294, i64 0, i64 0), i32 signext %conv4.i65.lcssa) #3
  unreachable

if.end23:                                         ; preds = %if.end14
  %conv15 = zext i32 %3 to i64
  %4 = load i32, i32* %numFreeResourcesInSubgroup, align 4
  %conv24 = zext i32 %4 to i64
  %5 = call { i64, i64, i64, i64 } asm sideeffect "sc", "=&{r0},=&{r3},=&{r4},=&{r5},{r0},{r3},{r4},{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{r12},~{cr0},~{memory}"(i64 1033, i64 0, i64 %0, i64 %inp64) #2
  %cmp5.1 = icmp eq i32 0, 0
  br i1 %cmp5.1, label %if.end.1, label %if.then

for.end38:                                        ; preds = %if.end23.3
  ret void

if.end.1:                                         ; preds = %if.end23
  %6 = load i32, i32* %numFreeResourcesInSubgroup, align 4
  %conv.1 = zext i32 %6 to i64
  %add.1 = add nuw nsw i64 %conv.1, %conv
  %7 = call { i64, i64, i64, i64 } asm sideeffect "sc", "=&{r0},=&{r3},=&{r4},=&{r5},{r0},{r3},{r4},{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{r12},~{cr0},~{memory}"(i64 1034, i64 0, i64 %0, i64 %inp64) #2
  %cmp10.1 = icmp eq i32 %inp32, 0
  br i1 %cmp10.1, label %if.end14.1, label %if.then11

if.end14.1:                                       ; preds = %if.end.1
  %8 = load i32, i32* %numFreeResourcesInSubgroup, align 4
  %cmp19.1 = icmp eq i32 0, 0
  br i1 %cmp19.1, label %if.end23.1, label %if.then20

if.end23.1:                                       ; preds = %if.end14.1
  %conv15.1 = zext i32 %8 to i64
  %add16.1 = add nuw nsw i64 %conv15.1, %conv15
  %9 = load i32, i32* %numFreeResourcesInSubgroup, align 4
  %conv24.1 = zext i32 %9 to i64
  %add25.1 = add nuw nsw i64 %conv24.1, %conv24
  %cmp5.2 = icmp eq i32 %inp32, 0
  br i1 %cmp5.2, label %if.end.2, label %if.then

if.end.2:                                         ; preds = %if.end23.1
  %10 = load i32, i32* %numFreeResourcesInSubgroup, align 4
  %conv.2 = zext i32 %10 to i64
  %add.2 = add nuw nsw i64 %conv.2, %add.1
  %11 = call { i64, i64, i64, i64 } asm sideeffect "sc", "=&{r0},=&{r3},=&{r4},=&{r5},{r0},{r3},{r4},{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{r12},~{cr0},~{memory}"(i64 1034, i64 %inp64, i64 %0, i64 %inp64) #2
  %cmp10.2 = icmp eq i32 0, 0
  br i1 %cmp10.2, label %if.end14.2, label %if.then11

if.end14.2:                                       ; preds = %if.end.2
  %12 = load i32, i32* %numFreeResourcesInSubgroup, align 4
  %13 = call { i64, i64, i64, i64 } asm sideeffect "sc", "=&{r0},=&{r3},=&{r4},=&{r5},{r0},{r3},{r4},{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{r12},~{cr0},~{memory}"(i64 1035, i64 %inp64, i64 %0, i64 0) #2
  %asmresult1.i64.2 = extractvalue { i64, i64, i64, i64 } %13, 1
  %conv4.i65.2 = trunc i64 %asmresult1.i64.2 to i32
  %cmp19.2 = icmp eq i32 %conv4.i65.2, 0
  br i1 %cmp19.2, label %if.end23.2, label %if.then20

if.end23.2:                                       ; preds = %if.end14.2
  %conv15.2 = zext i32 %12 to i64
  %add16.2 = add nuw nsw i64 %conv15.2, %add16.1
  %14 = load i32, i32* %numFreeResourcesInSubgroup, align 4
  %conv24.2 = zext i32 %14 to i64
  %add25.2 = add nuw nsw i64 %conv24.2, %add25.1
  %cmp5.3 = icmp eq i32 0, 0
  br i1 %cmp5.3, label %if.end.3, label %if.then

if.end.3:                                         ; preds = %if.end23.2
  %15 = load i32, i32* %numFreeResourcesInSubgroup, align 4
  %conv.3 = zext i32 %15 to i64
  %add.3 = add nuw nsw i64 %conv.3, %add.2
  %cmp10.3 = icmp eq i32 %inp32, 0
  br i1 %cmp10.3, label %if.end14.3, label %if.then11

if.end14.3:                                       ; preds = %if.end.3
  %16 = load i32, i32* %numFreeResourcesInSubgroup, align 4
  %17 = call { i64, i64, i64, i64 } asm sideeffect "sc", "=&{r0},=&{r3},=&{r4},=&{r5},{r0},{r3},{r4},{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{r12},~{cr0},~{memory}"(i64 1035, i64 0, i64 %0, i64 0) #2
  %asmresult1.i64.3 = extractvalue { i64, i64, i64, i64 } %17, 1
  %conv4.i65.3 = trunc i64 %asmresult1.i64.3 to i32
  %cmp19.3 = icmp eq i32 %conv4.i65.3, 0
  br i1 %cmp19.3, label %if.end23.3, label %if.then20

if.end23.3:                                       ; preds = %if.end14.3
  %conv15.3 = zext i32 %16 to i64
  %add16.3 = add nuw nsw i64 %conv15.3, %add16.2
  %add25.3 = add nuw nsw i64 0, %add25.2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 4
  %cmp27 = icmp ult i64 %add.3, %minFreeInjFifosPerCore.095
  %numFreeInjFifosInGroup.0.minFreeInjFifosPerCore.0 = select i1 %cmp27, i64 %add.3, i64 %minFreeInjFifosPerCore.095
  %cmp30 = icmp ult i64 %add16.3, %minFreeRecFifosPerCore.096
  %minFreeRecFifosPerCore.1 = select i1 %cmp30, i64 %add16.3, i64 %minFreeRecFifosPerCore.096
  %cmp33 = icmp ult i64 %add25.3, %minFreeBatIdsPerCore.097
  %numFreeBatIdsInGroup.0.minFreeBatIdsPerCore.0 = select i1 %cmp33, i64 %add25.3, i64 %minFreeBatIdsPerCore.097
  %inc37 = add nuw nsw i32 %group.098, 1
  %cmp = icmp ult i32 %inc37, 16
  br i1 %cmp, label %for.cond2.preheader, label %for.end38
}

attributes #0 = { nounwind "frame-pointer"="all" "target-cpu"="a2q" }
attributes #1 = { inlinehint nounwind "frame-pointer"="all" "target-cpu"="a2q" }
attributes #2 = { nounwind }
attributes #3 = { cold nounwind }

