; RUN: /home/tanmay/Documents/Tools/llvm-project/build/bin/opt -passes=error-analysis -S < %s | FileCheck %s

; ModuleID = '/home/tanmay/Documents/Tools/llvm-project/llvm/test/Transforms/ErrorAnalysis/LLVMInstructionsTest/select.c'
source_filename = "/home/tanmay/Documents/Tools/llvm-project/llvm/test/Transforms/ErrorAnalysis/LLVMInstructionsTest/select.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.stat = type { i64, i64, i64, i32, i32, i32, i32, i64, i64, i64, i64, %struct.timespec, %struct.timespec, %struct.timespec, [3 x i64] }
%struct.timespec = type { i64, i64 }
%struct.ACTable = type { i64, ptr }
%struct.ACItem = type { i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, i32 }
%struct.AFProduct = type { i32, ptr, ptr, ptr, i32, double }
%struct.AFTable = type { i64, ptr }
%struct.AFItem = type { i32, ptr, i32 }

@.str = private unnamed_addr constant [4 x i8] c"%lf\00", align 1
@.str.1 = private unnamed_addr constant [27 x i8] c"#fAC: Out of memory error!\00", align 1
@.str.7 = private unnamed_addr constant [6 x i8] c"(1.0)\00", align 1
@.str.9 = private unnamed_addr constant [5 x i8] c"cos(\00", align 1
@.str.10 = private unnamed_addr constant [3 x i8] c")/\00", align 1
@.str.11 = private unnamed_addr constant [5 x i8] c"sin(\00", align 1
@.str.12 = private unnamed_addr constant [5 x i8] c"tan(\00", align 1
@.str.13 = private unnamed_addr constant [3 x i8] c"/(\00", align 1
@.str.14 = private unnamed_addr constant [3 x i8] c")*\00", align 1
@.str.15 = private unnamed_addr constant [3 x i8] c"))\00", align 1
@.str.16 = private unnamed_addr constant [9 x i8] c"sqrt(1-(\00", align 1
@.str.17 = private unnamed_addr constant [11 x i8] c"))*arcsin(\00", align 1
@.str.18 = private unnamed_addr constant [3 x i8] c"(-\00", align 1
@.str.19 = private unnamed_addr constant [11 x i8] c"))*arccos(\00", align 1
@.str.20 = private unnamed_addr constant [5 x i8] c"/(((\00", align 1
@.str.21 = private unnamed_addr constant [13 x i8] c")+1)*arctan(\00", align 1
@.str.22 = private unnamed_addr constant [6 x i8] c"cosh(\00", align 1
@.str.23 = private unnamed_addr constant [6 x i8] c"sinh(\00", align 1
@.str.24 = private unnamed_addr constant [6 x i8] c"tanh(\00", align 1
@.str.25 = private unnamed_addr constant [7 x i8] c"1/log(\00", align 1
@.str.26 = private unnamed_addr constant [6 x i8] c"(0.5)\00", align 1
@.str.27 = private unnamed_addr constant [22 x i8] c"No such operation %d\0A\00", align 1
@.str.28 = private unnamed_addr constant [13 x i8] c"node-unknown\00", align 1
@.str.29 = private unnamed_addr constant [3 x i8] c"%d\00", align 1
@.str.32 = private unnamed_addr constant [55 x i8] c"File != NULL && \22Memory not allocated for File String\22\00", align 1
@.str.33 = private unnamed_addr constant [119 x i8] c"/home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AtomicCondition.h\00", align 1
@__PRETTY_FUNCTION__.fAFGenerateFileString = private unnamed_addr constant [63 x i8] c"void fAFGenerateFileString(char *, const char *, const char *)\00", align 1
@.str.34 = private unnamed_addr constant [10 x i8] c".fAF_logs\00", align 1
@ACs = dso_local local_unnamed_addr global ptr null, align 8
@.str.35 = private unnamed_addr constant [37 x i8] c"#fAC: Not enough memory for ACTable!\00", align 1
@.str.36 = private unnamed_addr constant [45 x i8] c"#fAC: Not enough memory for ACItem pointers!\00", align 1
@ACItemCounter = dso_local local_unnamed_addr global i32 0, align 4
@.str.37 = private unnamed_addr constant [36 x i8] c"#fAC: Not enough memory for ACItem!\00", align 1
@.str.38 = private unnamed_addr constant [42 x i8] c"#fAC: Not enough memory for OperandNames!\00", align 1
@.str.39 = private unnamed_addr constant [43 x i8] c"#fAC: Not enough memory for OperandValues!\00", align 1
@.str.40 = private unnamed_addr constant [33 x i8] c"#fAC: Not enough memory for ACs!\00", align 1
@.str.43 = private unnamed_addr constant [5 x i8] c"fAC_\00", align 1
@.str.44 = private unnamed_addr constant [6 x i8] c".json\00", align 1
@.str.45 = private unnamed_addr constant [2 x i8] c"w\00", align 1
@.str.46 = private unnamed_addr constant [3 x i8] c"{\0A\00", align 1
@.str.47 = private unnamed_addr constant [11 x i8] c"\09\22ACs\22: [\0A\00", align 1
@.str.48 = private unnamed_addr constant [63 x i8] c"\09\09{\0A\09\09\09\22ItemId\22: %d,\0A\09\09\09\22Function\22: %d,\0A\09\09\09\22ResultVar\22: \22%s\22,\0A\00", align 1
@.str.49 = private unnamed_addr constant [57 x i8] c"\09\09\09\22Operand %d Name\22: \22%s\22,\0A\09\09\09\22Operand %d Value\22: %lf,\0A\00", align 1
@.str.50 = private unnamed_addr constant [28 x i8] c"\09\09\09\22ACWRTOperand %d\22: %lf,\0A\00", align 1
@.str.51 = private unnamed_addr constant [30 x i8] c"\09\09\09\22ACStringWRTOp %d\22: \22%s\22,\0A\00", align 1
@.str.52 = private unnamed_addr constant [23 x i8] c"\09\09\09\22File Name\22: \22%s\22,\0A\00", align 1
@.str.53 = private unnamed_addr constant [22 x i8] c"\09\09\09\22Line Number\22: %d\0A\00", align 1
@.str.54 = private unnamed_addr constant [6 x i8] c"\09\09},\0A\00", align 1
@.str.55 = private unnamed_addr constant [5 x i8] c"\09\09}\0A\00", align 1
@.str.56 = private unnamed_addr constant [4 x i8] c"\09]\0A\00", align 1
@.str.57 = private unnamed_addr constant [3 x i8] c"}\0A\00", align 1
@.str.58 = private unnamed_addr constant [39 x i8] c"Atomic Conditions written to file: %s\0A\00", align 1
@fp32OpError = dso_local local_unnamed_addr global [19 x float] [float 0x3EB0C6F7A0000000, float 0x3EB0C6F7A0000000, float 0x3EB0C6F7A0000000, float 0x3EB0C6F7A0000000, float 0x3EB0C6F7A0000000, float 0x3EB0C6F7A0000000, float 0x3EB0C6F7A0000000, float 0x3EB0C6F7A0000000, float 0x3EB0C6F7A0000000, float 0x3EB0C6F7A0000000, float 0x3EC0C6F7A0000000, float 0x3EC0C6F7A0000000, float 0x3EC0C6F7A0000000, float 0x3EB0C6F7A0000000, float 0x3EB0C6F7A0000000, float 0x3EB0C6F7A0000000, float 0x3EB0C6F7A0000000, float 0x3EB0C6F7A0000000, float 0x3EB0C6F7A0000000], align 16
@fp64OpError = dso_local local_unnamed_addr global [19 x double] [double 0x3C9CD2B297D889BC, double 0x3C9CD2B297D889BC, double 0x3C9CD2B297D889BC, double 0x3C9CD2B297D889BC, double 0x3C9CD2B297D889BC, double 0x3C9CD2B297D889BC, double 0x3C9CD2B297D889BC, double 0x3C9CD2B297D889BC, double 0x3C9CD2B297D889BC, double 0x3C9CD2B297D889BC, double 2.000000e-16, double 2.000000e-16, double 2.000000e-16, double 0x3C9CD2B297D889BC, double 0x3C9CD2B297D889BC, double 0x3C9CD2B297D889BC, double 0x3C9CD2B297D889BC, double 0x3C9CD2B297D889BC, double 0x3C9CD2B297D889BC], align 16
@.str.59 = private unnamed_addr constant [39 x i8] c"#fAF: Not enough memory for AFProduct!\00", align 1
@.str.60 = private unnamed_addr constant [36 x i8] c"#fAF: Not enough memory for AFItem!\00", align 1
@.str.61 = private unnamed_addr constant [48 x i8] c"#fAF: Not enough memory for AFProduct pointers!\00", align 1
@AFComponentCounter = dso_local local_unnamed_addr global i32 0, align 4
@Paths = dso_local local_unnamed_addr global ptr null, align 8
@AFs = dso_local local_unnamed_addr global ptr null, align 8
@.str.62 = private unnamed_addr constant [5 x i8] c"load\00", align 1
@.str.63 = private unnamed_addr constant [7 x i8] c"alloca\00", align 1
@AFItemCounter = dso_local local_unnamed_addr global i32 0, align 4
@.str.64 = private unnamed_addr constant [45 x i8] c"#fAF: Not enough memory for AFItem pointers!\00", align 1
@.str.65 = private unnamed_addr constant [45 x i8] c"#fAF: Not enough memory for AFProduct Array!\00", align 1
@.str.69 = private unnamed_addr constant [47 x i8] c"AF: %f of Node:%d WRT Input:%s through path: [\00", align 1
@.str.70 = private unnamed_addr constant [5 x i8] c", %d\00", align 1
@.str.75 = private unnamed_addr constant [5 x i8] c"fAF_\00", align 1
@.str.76 = private unnamed_addr constant [11 x i8] c"\09\22AFs\22: [\0A\00", align 1
@.str.77 = private unnamed_addr constant [132 x i8] c"\09\09{\0A\09\09\09\22ProductItemId\22: %d,\0A\09\09\09\22ACItemId\22:%d,\0A\09\09\09\22ACItemString\22: \22%s\22,\0A\09\09\09\22ProductTailItemId\22: %d,\0A\09\09\09\22Input\22: \22%s\22,\0A\09\09\09\22AF\22: %lf,\0A\00", align 1
@.str.78 = private unnamed_addr constant [29 x i8] c"\09\09\09\22Path(AFProductIds)\22: [%d\00", align 1
@.str.79 = private unnamed_addr constant [7 x i8] c"]\0A\09\09}\0A\00", align 1
@.str.80 = private unnamed_addr constant [8 x i8] c"]\0A\09\09},\0A\00", align 1
@.str.81 = private unnamed_addr constant [43 x i8] c"Amplification Factors written to file: %s\0A\00", align 1
@a = dso_local local_unnamed_addr global i32 0, align 4
@str = private unnamed_addr constant [18 x i8] c"No such operation\00", align 1
@str.82 = private unnamed_addr constant [36 x i8] c"\0AWriting Atomic Conditions to file.\00", align 1
@str.83 = private unnamed_addr constant [50 x i8] c"Printing Top Amplification Paths from Last AFItem\00", align 1
@str.87 = private unnamed_addr constant [48 x i8] c"Printing Top Amplification Paths Over ALL Paths\00", align 1
@str.88 = private unnamed_addr constant [33 x i8] c"The top Amplification Paths are:\00", align 1
@str.89 = private unnamed_addr constant [32 x i8] c"Printed Top Amplification Paths\00", align 1
@str.90 = private unnamed_addr constant [2 x i8] c"]\00", align 1
@str.91 = private unnamed_addr constant [40 x i8] c"\0AWriting Amplification Factors to file.\00", align 1

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn uwtable
define dso_local i32 @fACIsTernaryOperation(i32 noundef %F) local_unnamed_addr #0 {
entry:
  %cmp = icmp eq i32 %F, 17
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn uwtable
define dso_local i32 @fACIsBinaryOperation(i32 noundef %F) local_unnamed_addr #0 {
entry:
  %0 = icmp ult i32 %F, 4
  %1 = zext i1 %0 to i32
  ret i32 %1
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn uwtable
define dso_local i32 @fACIsUnaryOperation(i32 noundef %F) local_unnamed_addr #0 {
entry:
  %F.off = add i32 %F, -4
  %switch = icmp ult i32 %F.off, 13
  %. = zext i1 %switch to i32
  ret i32 %.
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn uwtable
define dso_local i32 @fACFuncHasXNumOperands(i32 noundef %F) local_unnamed_addr #0 {
entry:
  %0 = add i32 %F, -17
  %switch.i = icmp ult i32 %0, -13
  br i1 %switch.i, label %if.end, label %return

if.end:                                           ; preds = %entry
  %1 = icmp ugt i32 %F, 3
  br i1 %1, label %if.end4, label %return

if.end4:                                          ; preds = %if.end
  %cmp.i.not = icmp eq i32 %F, 17
  %. = select i1 %cmp.i.not, i32 3, i32 0
  br label %return

return:                                           ; preds = %if.end4, %if.end, %entry
  %retval.0 = phi i32 [ 1, %entry ], [ 2, %if.end ], [ %., %if.end4 ]
  ret i32 %retval.0
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @fACAppendDoubleToString(ptr noundef %String, double noundef %DoubleValue) local_unnamed_addr #1 {
entry:
  %DoubleBuffer = alloca [15 x i8], align 1
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer) #21
  %call = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer, i64 noundef 15, ptr noundef nonnull @.str, double noundef %DoubleValue) #21
  %call2 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %String, ptr noundef nonnull %DoubleBuffer) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer) #21
  ret void
}

; Function Attrs: argmemonly mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #2

; Function Attrs: nofree nounwind
declare noundef i32 @snprintf(ptr noalias nocapture noundef writeonly, i64 noundef, ptr nocapture noundef readonly, ...) local_unnamed_addr #3

; Function Attrs: argmemonly mustprogress nofree nounwind willreturn
declare ptr @strcat(ptr noalias noundef returned, ptr noalias nocapture noundef readonly) local_unnamed_addr #4

; Function Attrs: argmemonly mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #2

; Function Attrs: nounwind uwtable
define dso_local ptr @fACDumpAtomicConditionString(ptr noundef readonly %OperandNames, ptr noundef readonly %OperandValues, i32 noundef %F, i32 noundef %WRT) local_unnamed_addr #5 {
entry:
  %DoubleBuffer.i1240 = alloca [15 x i8], align 1
  %DoubleBuffer.i1237 = alloca [15 x i8], align 1
  %DoubleBuffer.i1234 = alloca [15 x i8], align 1
  %DoubleBuffer.i1231 = alloca [15 x i8], align 1
  %DoubleBuffer.i1228 = alloca [15 x i8], align 1
  %DoubleBuffer.i1225 = alloca [15 x i8], align 1
  %DoubleBuffer.i1222 = alloca [15 x i8], align 1
  %DoubleBuffer.i1219 = alloca [15 x i8], align 1
  %DoubleBuffer.i1216 = alloca [15 x i8], align 1
  %DoubleBuffer.i1213 = alloca [15 x i8], align 1
  %DoubleBuffer.i1210 = alloca [15 x i8], align 1
  %DoubleBuffer.i1207 = alloca [15 x i8], align 1
  %DoubleBuffer.i1204 = alloca [15 x i8], align 1
  %DoubleBuffer.i1201 = alloca [15 x i8], align 1
  %DoubleBuffer.i1198 = alloca [15 x i8], align 1
  %DoubleBuffer.i1195 = alloca [15 x i8], align 1
  %DoubleBuffer.i1192 = alloca [15 x i8], align 1
  %DoubleBuffer.i1189 = alloca [15 x i8], align 1
  %DoubleBuffer.i1186 = alloca [15 x i8], align 1
  %DoubleBuffer.i1183 = alloca [15 x i8], align 1
  %DoubleBuffer.i1180 = alloca [15 x i8], align 1
  %DoubleBuffer.i1177 = alloca [15 x i8], align 1
  %DoubleBuffer.i1174 = alloca [15 x i8], align 1
  %DoubleBuffer.i1171 = alloca [15 x i8], align 1
  %DoubleBuffer.i1168 = alloca [15 x i8], align 1
  %DoubleBuffer.i1165 = alloca [15 x i8], align 1
  %DoubleBuffer.i1162 = alloca [15 x i8], align 1
  %DoubleBuffer.i1159 = alloca [15 x i8], align 1
  %DoubleBuffer.i1156 = alloca [15 x i8], align 1
  %DoubleBuffer.i1153 = alloca [15 x i8], align 1
  %DoubleBuffer.i1150 = alloca [15 x i8], align 1
  %DoubleBuffer.i1147 = alloca [15 x i8], align 1
  %DoubleBuffer.i1144 = alloca [15 x i8], align 1
  %DoubleBuffer.i1141 = alloca [15 x i8], align 1
  %DoubleBuffer.i = alloca [15 x i8], align 1
  %call = call noalias dereferenceable_or_null(150) ptr @malloc(i64 noundef 150) #22
  %cmp = icmp eq ptr %call, null
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call1 = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.1)
  call void @exit(i32 noundef 1) #23
  unreachable

if.end:                                           ; preds = %entry
  store i8 0, ptr %call, align 1, !tbaa !5
  switch i32 %F, label %sw.default [
    i32 0, label %sw.bb
    i32 1, label %sw.bb50
    i32 2, label %sw.bb133
    i32 3, label %sw.bb133
    i32 4, label %sw.bb135
    i32 5, label %sw.bb170
    i32 6, label %sw.bb194
    i32 7, label %sw.bb229
    i32 8, label %sw.bb273
    i32 9, label %sw.bb317
    i32 10, label %sw.bb360
    i32 11, label %sw.bb395
    i32 12, label %sw.bb419
    i32 13, label %sw.bb454
    i32 14, label %sw.bb466
    i32 15, label %sw.bb479
    i32 16, label %sw.bb481
    i32 17, label %sw.bb483
  ]

sw.bb:                                            ; preds = %if.end
  switch i32 %WRT, label %if.end26 [
    i32 0, label %if.then2
    i32 1, label %if.then14
  ]

if.then2:                                         ; preds = %sw.bb
  %strlen1104 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1105 = getelementptr i8, ptr %call, i64 %strlen1104
  store i16 40, ptr %endptr1105, align 1
  %0 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char01106 = load i8, ptr %0, align 1
  %cmp6.not = icmp eq i8 %char01106, 0
  br i1 %cmp6.not, label %if.else, label %if.then7

if.then7:                                         ; preds = %if.then2
  %call9 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %0) #21
  br label %if.end26

if.else:                                          ; preds = %if.then2
  %1 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @fACAppendDoubleToString(ptr noundef %call, double noundef %1)
  br label %if.end26

if.then14:                                        ; preds = %sw.bb
  %arrayidx15 = getelementptr inbounds ptr, ptr %OperandNames, i64 1
  %2 = load ptr, ptr %arrayidx15, align 8, !tbaa !8
  %char01119 = load i8, ptr %2, align 1
  %cmp17.not = icmp eq i8 %char01119, 0
  br i1 %cmp17.not, label %if.else22, label %if.then18

if.then18:                                        ; preds = %if.then14
  %strlen1120 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1121 = getelementptr i8, ptr %call, i64 %strlen1120
  store i16 40, ptr %endptr1121, align 1
  %call21 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull dereferenceable(1) %2) #21
  br label %if.end26

if.else22:                                        ; preds = %if.then14
  %arrayidx23 = getelementptr inbounds double, ptr %OperandValues, i64 1
  %3 = load double, ptr %arrayidx23, align 8, !tbaa !10
  call void @fACAppendDoubleToString(ptr noundef %call, double noundef %3)
  br label %if.end26

if.end26:                                         ; preds = %sw.bb, %if.else22, %if.then18, %if.then7, %if.else
  %strlen1107 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1108 = getelementptr i8, ptr %call, i64 %strlen1107
  store i16 47, ptr %endptr1108, align 1
  %strlen1109 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1110 = getelementptr i8, ptr %call, i64 %strlen1109
  store i16 40, ptr %endptr1110, align 1
  %4 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char01111 = load i8, ptr %4, align 1
  %cmp31.not = icmp eq i8 %char01111, 0
  br i1 %cmp31.not, label %if.else35, label %if.then32

if.then32:                                        ; preds = %if.end26
  %call34 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %4) #21
  br label %if.end37

if.else35:                                        ; preds = %if.end26
  %5 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i) #21
  %call.i = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i, i64 noundef 15, ptr noundef nonnull @.str, double noundef %5) #21
  %call2.i = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i) #21
  br label %if.end37

if.end37:                                         ; preds = %if.else35, %if.then32
  %strlen1112 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1113 = getelementptr i8, ptr %call, i64 %strlen1112
  store i16 43, ptr %endptr1113, align 1
  %arrayidx39 = getelementptr inbounds ptr, ptr %OperandNames, i64 1
  %6 = load ptr, ptr %arrayidx39, align 8, !tbaa !8
  %char01114 = load i8, ptr %6, align 1
  %cmp41.not = icmp eq i8 %char01114, 0
  br i1 %cmp41.not, label %if.else45, label %if.then42

if.then42:                                        ; preds = %if.end37
  %call44 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %6) #21
  br label %if.end47

if.else45:                                        ; preds = %if.end37
  %arrayidx46 = getelementptr inbounds double, ptr %OperandValues, i64 1
  %7 = load double, ptr %arrayidx46, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1141) #21
  %call.i1142 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1141, i64 noundef 15, ptr noundef nonnull @.str, double noundef %7) #21
  %call2.i1143 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1141) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1141) #21
  br label %if.end47

if.end47:                                         ; preds = %if.else45, %if.then42
  %strlen1115 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1116 = getelementptr i8, ptr %call, i64 %strlen1115
  store i16 41, ptr %endptr1116, align 1
  %strlen1117 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1118 = getelementptr i8, ptr %call, i64 %strlen1117
  store i16 41, ptr %endptr1118, align 1
  br label %sw.epilog

sw.bb50:                                          ; preds = %if.end
  switch i32 %WRT, label %if.else91.critedge [
    i32 0, label %if.then52
    i32 1, label %if.then65
  ]

if.then52:                                        ; preds = %sw.bb50
  %8 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char01084 = load i8, ptr %8, align 1
  %cmp55.not = icmp eq i8 %char01084, 0
  br i1 %cmp55.not, label %if.else60, label %if.then56

if.then56:                                        ; preds = %if.then52
  %strlen1099 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1100 = getelementptr i8, ptr %call, i64 %strlen1099
  store i16 40, ptr %endptr1100, align 1
  %call59 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull dereferenceable(1) %8) #21
  br label %if.then81

if.else60:                                        ; preds = %if.then52
  %9 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @fACAppendDoubleToString(ptr noundef %call, double noundef %9)
  br label %if.then81

if.then65:                                        ; preds = %sw.bb50
  %arrayidx66 = getelementptr inbounds ptr, ptr %OperandNames, i64 1
  %10 = load ptr, ptr %arrayidx66, align 8, !tbaa !8
  %char01101 = load i8, ptr %10, align 1
  %cmp68.not = icmp eq i8 %char01101, 0
  br i1 %cmp68.not, label %if.else73, label %if.then69

if.then69:                                        ; preds = %if.then65
  %strlen1102 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1103 = getelementptr i8, ptr %call, i64 %strlen1102
  store i16 40, ptr %endptr1103, align 1
  %call72 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull dereferenceable(1) %10) #21
  br label %if.then93

if.else73:                                        ; preds = %if.then65
  %arrayidx74 = getelementptr inbounds double, ptr %OperandValues, i64 1
  %11 = load double, ptr %arrayidx74, align 8, !tbaa !10
  call void @fACAppendDoubleToString(ptr noundef %call, double noundef %11)
  br label %if.then93

if.then81:                                        ; preds = %if.else60, %if.then56
  %strlen1085.c1137 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1086.c1138 = getelementptr i8, ptr %call, i64 %strlen1085.c1137
  store i16 47, ptr %endptr1086.c1138, align 1
  %strlen1087.c1139 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1088.c1140 = getelementptr i8, ptr %call, i64 %strlen1087.c1139
  store i16 40, ptr %endptr1088.c1140, align 1
  %12 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char01089 = load i8, ptr %12, align 1
  %cmp84.not = icmp eq i8 %char01089, 0
  br i1 %cmp84.not, label %if.else88, label %if.then85

if.then85:                                        ; preds = %if.then81
  %call87 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %12) #21
  br label %if.then107

if.else88:                                        ; preds = %if.then81
  %13 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @fACAppendDoubleToString(ptr noundef %call, double noundef %13)
  br label %if.then107

if.else91.critedge:                               ; preds = %sw.bb50
  %strlen1085.c = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1086.c = getelementptr i8, ptr %call, i64 %strlen1085.c
  store i16 47, ptr %endptr1086.c, align 1
  %strlen1087.c = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1088.c = getelementptr i8, ptr %call, i64 %strlen1087.c
  store i16 40, ptr %endptr1088.c, align 1
  br label %if.else117

if.then93:                                        ; preds = %if.else73, %if.then69
  %strlen1085.c1123 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1086.c1124 = getelementptr i8, ptr %call, i64 %strlen1085.c1123
  store i16 47, ptr %endptr1086.c1124, align 1
  %strlen1087.c1125 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1088.c1126 = getelementptr i8, ptr %call, i64 %strlen1087.c1125
  store i16 40, ptr %endptr1088.c1126, align 1
  %arrayidx94 = getelementptr inbounds ptr, ptr %OperandNames, i64 1
  %14 = load ptr, ptr %arrayidx94, align 8, !tbaa !8
  %char01098 = load i8, ptr %14, align 1
  %cmp96.not = icmp eq i8 %char01098, 0
  br i1 %cmp96.not, label %if.else100, label %if.then97

if.then97:                                        ; preds = %if.then93
  %call99 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %14) #21
  br label %if.else117

if.else100:                                       ; preds = %if.then93
  %arrayidx101 = getelementptr inbounds double, ptr %OperandValues, i64 1
  %15 = load double, ptr %arrayidx101, align 8, !tbaa !10
  call void @fACAppendDoubleToString(ptr noundef %call, double noundef %15)
  br label %if.else117

if.then107:                                       ; preds = %if.else88, %if.then85
  %strlen1090.c1252 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1091.c1253 = getelementptr i8, ptr %call, i64 %strlen1090.c1252
  store i16 45, ptr %endptr1091.c1253, align 1
  %arrayidx108 = getelementptr inbounds ptr, ptr %OperandNames, i64 1
  %16 = load ptr, ptr %arrayidx108, align 8, !tbaa !8
  %char01092 = load i8, ptr %16, align 1
  %cmp110.not = icmp eq i8 %char01092, 0
  br i1 %cmp110.not, label %if.else114, label %if.then111

if.then111:                                       ; preds = %if.then107
  %call113 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %16) #21
  br label %if.end130

if.else114:                                       ; preds = %if.then107
  %arrayidx115 = getelementptr inbounds double, ptr %OperandValues, i64 1
  %17 = load double, ptr %arrayidx115, align 8, !tbaa !10
  call void @fACAppendDoubleToString(ptr noundef %call, double noundef %17)
  br label %if.end130

if.else117:                                       ; preds = %if.then97, %if.else100, %if.else91.critedge
  %strlen1090.c1247 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1091.c1248 = getelementptr i8, ptr %call, i64 %strlen1090.c1247
  store i16 45, ptr %endptr1091.c1248, align 1
  %cmp118 = icmp eq i32 %WRT, 1
  br i1 %cmp118, label %if.then119, label %if.end130

if.then119:                                       ; preds = %if.else117
  %18 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char01097 = load i8, ptr %18, align 1
  %cmp122.not = icmp eq i8 %char01097, 0
  br i1 %cmp122.not, label %if.else126, label %if.then123

if.then123:                                       ; preds = %if.then119
  %call125 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %18) #21
  br label %if.end130

if.else126:                                       ; preds = %if.then119
  %19 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @fACAppendDoubleToString(ptr noundef %call, double noundef %19)
  br label %if.end130

if.end130:                                        ; preds = %if.else117, %if.else126, %if.then123, %if.then111, %if.else114
  %strlen1093 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1094 = getelementptr i8, ptr %call, i64 %strlen1093
  store i16 41, ptr %endptr1094, align 1
  %strlen1095 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1096 = getelementptr i8, ptr %call, i64 %strlen1095
  store i16 41, ptr %endptr1096, align 1
  br label %sw.epilog

sw.bb133:                                         ; preds = %if.end, %if.end
  %strlen1082 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1083 = getelementptr i8, ptr %call, i64 %strlen1082
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(6) %endptr1083, ptr noundef nonnull align 1 dereferenceable(6) @.str.7, i64 6, i1 false)
  br label %sw.epilog

sw.bb135:                                         ; preds = %if.end
  %strlen1065 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1066 = getelementptr i8, ptr %call, i64 %strlen1065
  store i16 40, ptr %endptr1066, align 1
  %20 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char01067 = load i8, ptr %20, align 1
  %cmp139.not = icmp eq i8 %char01067, 0
  br i1 %cmp139.not, label %if.else143, label %if.then140

if.then140:                                       ; preds = %sw.bb135
  %call142 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %20) #21
  br label %if.end145

if.else143:                                       ; preds = %sw.bb135
  %21 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1144) #21
  %call.i1145 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1144, i64 noundef 15, ptr noundef nonnull @.str, double noundef %21) #21
  %call2.i1146 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1144) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1144) #21
  br label %if.end145

if.end145:                                        ; preds = %if.else143, %if.then140
  %strlen1068 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1069 = getelementptr i8, ptr %call, i64 %strlen1068
  store i16 42, ptr %endptr1069, align 1
  %strlen1070 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1071 = getelementptr i8, ptr %call, i64 %strlen1070
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(5) %endptr1071, ptr noundef nonnull align 1 dereferenceable(5) @.str.9, i64 5, i1 false)
  %22 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char01072 = load i8, ptr %22, align 1
  %cmp150.not = icmp eq i8 %char01072, 0
  br i1 %cmp150.not, label %if.else154, label %if.then151

if.then151:                                       ; preds = %if.end145
  %call153 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %22) #21
  br label %if.end156

if.else154:                                       ; preds = %if.end145
  %23 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1147) #21
  %call.i1148 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1147, i64 noundef 15, ptr noundef nonnull @.str, double noundef %23) #21
  %call2.i1149 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1147) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1147) #21
  br label %if.end156

if.end156:                                        ; preds = %if.else154, %if.then151
  %strlen1073 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1074 = getelementptr i8, ptr %call, i64 %strlen1073
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(3) %endptr1074, ptr noundef nonnull align 1 dereferenceable(3) @.str.10, i64 3, i1 false)
  %strlen1075 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1076 = getelementptr i8, ptr %call, i64 %strlen1075
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(5) %endptr1076, ptr noundef nonnull align 1 dereferenceable(5) @.str.11, i64 5, i1 false)
  %24 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char01077 = load i8, ptr %24, align 1
  %cmp161.not = icmp eq i8 %char01077, 0
  br i1 %cmp161.not, label %if.else165, label %if.then162

if.then162:                                       ; preds = %if.end156
  %call164 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %24) #21
  br label %if.end167

if.else165:                                       ; preds = %if.end156
  %25 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1150) #21
  %call.i1151 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1150, i64 noundef 15, ptr noundef nonnull @.str, double noundef %25) #21
  %call2.i1152 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1150) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1150) #21
  br label %if.end167

if.end167:                                        ; preds = %if.else165, %if.then162
  %strlen1078 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1079 = getelementptr i8, ptr %call, i64 %strlen1078
  store i16 41, ptr %endptr1079, align 1
  %strlen1080 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1081 = getelementptr i8, ptr %call, i64 %strlen1080
  store i16 41, ptr %endptr1081, align 1
  br label %sw.epilog

sw.bb170:                                         ; preds = %if.end
  %strlen1053 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1054 = getelementptr i8, ptr %call, i64 %strlen1053
  store i16 40, ptr %endptr1054, align 1
  %26 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char01055 = load i8, ptr %26, align 1
  %cmp174.not = icmp eq i8 %char01055, 0
  br i1 %cmp174.not, label %if.else178, label %if.then175

if.then175:                                       ; preds = %sw.bb170
  %call177 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %26) #21
  br label %if.end180

if.else178:                                       ; preds = %sw.bb170
  %27 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1153) #21
  %call.i1154 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1153, i64 noundef 15, ptr noundef nonnull @.str, double noundef %27) #21
  %call2.i1155 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1153) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1153) #21
  br label %if.end180

if.end180:                                        ; preds = %if.else178, %if.then175
  %strlen1056 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1057 = getelementptr i8, ptr %call, i64 %strlen1056
  store i16 42, ptr %endptr1057, align 1
  %strlen1058 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1059 = getelementptr i8, ptr %call, i64 %strlen1058
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(5) %endptr1059, ptr noundef nonnull align 1 dereferenceable(5) @.str.12, i64 5, i1 false)
  %28 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char01060 = load i8, ptr %28, align 1
  %cmp185.not = icmp eq i8 %char01060, 0
  br i1 %cmp185.not, label %if.else189, label %if.then186

if.then186:                                       ; preds = %if.end180
  %call188 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %28) #21
  br label %if.end191

if.else189:                                       ; preds = %if.end180
  %29 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1156) #21
  %call.i1157 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1156, i64 noundef 15, ptr noundef nonnull @.str, double noundef %29) #21
  %call2.i1158 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1156) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1156) #21
  br label %if.end191

if.end191:                                        ; preds = %if.else189, %if.then186
  %strlen1061 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1062 = getelementptr i8, ptr %call, i64 %strlen1061
  store i16 41, ptr %endptr1062, align 1
  %strlen1063 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1064 = getelementptr i8, ptr %call, i64 %strlen1063
  store i16 41, ptr %endptr1064, align 1
  br label %sw.epilog

sw.bb194:                                         ; preds = %if.end
  %strlen1036 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1037 = getelementptr i8, ptr %call, i64 %strlen1036
  store i16 40, ptr %endptr1037, align 1
  %30 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char01038 = load i8, ptr %30, align 1
  %cmp198.not = icmp eq i8 %char01038, 0
  br i1 %cmp198.not, label %if.else202, label %if.then199

if.then199:                                       ; preds = %sw.bb194
  %call201 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %30) #21
  br label %if.end204

if.else202:                                       ; preds = %sw.bb194
  %31 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1159) #21
  %call.i1160 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1159, i64 noundef 15, ptr noundef nonnull @.str, double noundef %31) #21
  %call2.i1161 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1159) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1159) #21
  br label %if.end204

if.end204:                                        ; preds = %if.else202, %if.then199
  %strlen1039 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1040 = getelementptr i8, ptr %call, i64 %strlen1039
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(3) %endptr1040, ptr noundef nonnull align 1 dereferenceable(3) @.str.13, i64 3, i1 false)
  %strlen1041 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1042 = getelementptr i8, ptr %call, i64 %strlen1041
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(5) %endptr1042, ptr noundef nonnull align 1 dereferenceable(5) @.str.11, i64 5, i1 false)
  %32 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char01043 = load i8, ptr %32, align 1
  %cmp209.not = icmp eq i8 %char01043, 0
  br i1 %cmp209.not, label %if.else213, label %if.then210

if.then210:                                       ; preds = %if.end204
  %call212 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %32) #21
  br label %if.end215

if.else213:                                       ; preds = %if.end204
  %33 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1162) #21
  %call.i1163 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1162, i64 noundef 15, ptr noundef nonnull @.str, double noundef %33) #21
  %call2.i1164 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1162) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1162) #21
  br label %if.end215

if.end215:                                        ; preds = %if.else213, %if.then210
  %strlen1044 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1045 = getelementptr i8, ptr %call, i64 %strlen1044
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(3) %endptr1045, ptr noundef nonnull align 1 dereferenceable(3) @.str.14, i64 3, i1 false)
  %strlen1046 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1047 = getelementptr i8, ptr %call, i64 %strlen1046
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(5) %endptr1047, ptr noundef nonnull align 1 dereferenceable(5) @.str.9, i64 5, i1 false)
  %34 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char01048 = load i8, ptr %34, align 1
  %cmp220.not = icmp eq i8 %char01048, 0
  br i1 %cmp220.not, label %if.else224, label %if.then221

if.then221:                                       ; preds = %if.end215
  %call223 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %34) #21
  br label %if.end226

if.else224:                                       ; preds = %if.end215
  %35 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1165) #21
  %call.i1166 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1165, i64 noundef 15, ptr noundef nonnull @.str, double noundef %35) #21
  %call2.i1167 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1165) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1165) #21
  br label %if.end226

if.end226:                                        ; preds = %if.else224, %if.then221
  %strlen1049 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1050 = getelementptr i8, ptr %call, i64 %strlen1049
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(3) %endptr1050, ptr noundef nonnull align 1 dereferenceable(3) @.str.15, i64 3, i1 false)
  %strlen1051 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1052 = getelementptr i8, ptr %call, i64 %strlen1051
  store i16 41, ptr %endptr1052, align 1
  br label %sw.epilog

sw.bb229:                                         ; preds = %if.end
  %strlen1018 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1019 = getelementptr i8, ptr %call, i64 %strlen1018
  store i16 40, ptr %endptr1019, align 1
  %36 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char01020 = load i8, ptr %36, align 1
  %cmp233.not = icmp eq i8 %char01020, 0
  br i1 %cmp233.not, label %if.else237, label %if.then234

if.then234:                                       ; preds = %sw.bb229
  %call236 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %36) #21
  br label %if.end239

if.else237:                                       ; preds = %sw.bb229
  %37 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1168) #21
  %call.i1169 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1168, i64 noundef 15, ptr noundef nonnull @.str, double noundef %37) #21
  %call2.i1170 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1168) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1168) #21
  br label %if.end239

if.end239:                                        ; preds = %if.else237, %if.then234
  %strlen1021 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1022 = getelementptr i8, ptr %call, i64 %strlen1021
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(3) %endptr1022, ptr noundef nonnull align 1 dereferenceable(3) @.str.13, i64 3, i1 false)
  %strlen1023 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1024 = getelementptr i8, ptr %call, i64 %strlen1023
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(9) %endptr1024, ptr noundef nonnull align 1 dereferenceable(9) @.str.16, i64 9, i1 false)
  %38 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char01025 = load i8, ptr %38, align 1
  %cmp244.not = icmp eq i8 %char01025, 0
  br i1 %cmp244.not, label %if.else248, label %if.then245

if.then245:                                       ; preds = %if.end239
  %call247 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %38) #21
  br label %if.end250

if.else248:                                       ; preds = %if.end239
  %39 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1171) #21
  %call.i1172 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1171, i64 noundef 15, ptr noundef nonnull @.str, double noundef %39) #21
  %call2.i1173 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1171) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1171) #21
  br label %if.end250

if.end250:                                        ; preds = %if.else248, %if.then245
  %strlen1026 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1027 = getelementptr i8, ptr %call, i64 %strlen1026
  store i16 42, ptr %endptr1027, align 1
  %40 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char01028 = load i8, ptr %40, align 1
  %cmp254.not = icmp eq i8 %char01028, 0
  br i1 %cmp254.not, label %if.else258, label %if.then255

if.then255:                                       ; preds = %if.end250
  %call257 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %40) #21
  br label %if.end260

if.else258:                                       ; preds = %if.end250
  %41 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1174) #21
  %call.i1175 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1174, i64 noundef 15, ptr noundef nonnull @.str, double noundef %41) #21
  %call2.i1176 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1174) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1174) #21
  br label %if.end260

if.end260:                                        ; preds = %if.else258, %if.then255
  %strlen1029 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1030 = getelementptr i8, ptr %call, i64 %strlen1029
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(11) %endptr1030, ptr noundef nonnull align 1 dereferenceable(11) @.str.17, i64 11, i1 false)
  %42 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char01031 = load i8, ptr %42, align 1
  %cmp264.not = icmp eq i8 %char01031, 0
  br i1 %cmp264.not, label %if.else268, label %if.then265

if.then265:                                       ; preds = %if.end260
  %call267 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %42) #21
  br label %if.end270

if.else268:                                       ; preds = %if.end260
  %43 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1177) #21
  %call.i1178 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1177, i64 noundef 15, ptr noundef nonnull @.str, double noundef %43) #21
  %call2.i1179 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1177) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1177) #21
  br label %if.end270

if.end270:                                        ; preds = %if.else268, %if.then265
  %strlen1032 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1033 = getelementptr i8, ptr %call, i64 %strlen1032
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(3) %endptr1033, ptr noundef nonnull align 1 dereferenceable(3) @.str.15, i64 3, i1 false)
  %strlen1034 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1035 = getelementptr i8, ptr %call, i64 %strlen1034
  store i16 41, ptr %endptr1035, align 1
  br label %sw.epilog

sw.bb273:                                         ; preds = %if.end
  %strlen1000 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1001 = getelementptr i8, ptr %call, i64 %strlen1000
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(3) %endptr1001, ptr noundef nonnull align 1 dereferenceable(3) @.str.18, i64 3, i1 false)
  %44 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char01002 = load i8, ptr %44, align 1
  %cmp277.not = icmp eq i8 %char01002, 0
  br i1 %cmp277.not, label %if.else281, label %if.then278

if.then278:                                       ; preds = %sw.bb273
  %call280 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %44) #21
  br label %if.end283

if.else281:                                       ; preds = %sw.bb273
  %45 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1180) #21
  %call.i1181 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1180, i64 noundef 15, ptr noundef nonnull @.str, double noundef %45) #21
  %call2.i1182 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1180) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1180) #21
  br label %if.end283

if.end283:                                        ; preds = %if.else281, %if.then278
  %strlen1003 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1004 = getelementptr i8, ptr %call, i64 %strlen1003
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(3) %endptr1004, ptr noundef nonnull align 1 dereferenceable(3) @.str.13, i64 3, i1 false)
  %strlen1005 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1006 = getelementptr i8, ptr %call, i64 %strlen1005
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(9) %endptr1006, ptr noundef nonnull align 1 dereferenceable(9) @.str.16, i64 9, i1 false)
  %46 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char01007 = load i8, ptr %46, align 1
  %cmp288.not = icmp eq i8 %char01007, 0
  br i1 %cmp288.not, label %if.else292, label %if.then289

if.then289:                                       ; preds = %if.end283
  %call291 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %46) #21
  br label %if.end294

if.else292:                                       ; preds = %if.end283
  %47 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1183) #21
  %call.i1184 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1183, i64 noundef 15, ptr noundef nonnull @.str, double noundef %47) #21
  %call2.i1185 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1183) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1183) #21
  br label %if.end294

if.end294:                                        ; preds = %if.else292, %if.then289
  %strlen1008 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1009 = getelementptr i8, ptr %call, i64 %strlen1008
  store i16 42, ptr %endptr1009, align 1
  %48 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char01010 = load i8, ptr %48, align 1
  %cmp298.not = icmp eq i8 %char01010, 0
  br i1 %cmp298.not, label %if.else302, label %if.then299

if.then299:                                       ; preds = %if.end294
  %call301 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %48) #21
  br label %if.end304

if.else302:                                       ; preds = %if.end294
  %49 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1186) #21
  %call.i1187 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1186, i64 noundef 15, ptr noundef nonnull @.str, double noundef %49) #21
  %call2.i1188 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1186) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1186) #21
  br label %if.end304

if.end304:                                        ; preds = %if.else302, %if.then299
  %strlen1011 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1012 = getelementptr i8, ptr %call, i64 %strlen1011
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(11) %endptr1012, ptr noundef nonnull align 1 dereferenceable(11) @.str.19, i64 11, i1 false)
  %50 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char01013 = load i8, ptr %50, align 1
  %cmp308.not = icmp eq i8 %char01013, 0
  br i1 %cmp308.not, label %if.else312, label %if.then309

if.then309:                                       ; preds = %if.end304
  %call311 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %50) #21
  br label %if.end314

if.else312:                                       ; preds = %if.end304
  %51 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1189) #21
  %call.i1190 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1189, i64 noundef 15, ptr noundef nonnull @.str, double noundef %51) #21
  %call2.i1191 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1189) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1189) #21
  br label %if.end314

if.end314:                                        ; preds = %if.else312, %if.then309
  %strlen1014 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1015 = getelementptr i8, ptr %call, i64 %strlen1014
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(3) %endptr1015, ptr noundef nonnull align 1 dereferenceable(3) @.str.15, i64 3, i1 false)
  %strlen1016 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr1017 = getelementptr i8, ptr %call, i64 %strlen1016
  store i16 41, ptr %endptr1017, align 1
  br label %sw.epilog

sw.bb317:                                         ; preds = %if.end
  %strlen984 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr985 = getelementptr i8, ptr %call, i64 %strlen984
  store i16 40, ptr %endptr985, align 1
  %52 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char0986 = load i8, ptr %52, align 1
  %cmp321.not = icmp eq i8 %char0986, 0
  br i1 %cmp321.not, label %if.else325, label %if.then322

if.then322:                                       ; preds = %sw.bb317
  %call324 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %52) #21
  br label %if.end327

if.else325:                                       ; preds = %sw.bb317
  %53 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1192) #21
  %call.i1193 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1192, i64 noundef 15, ptr noundef nonnull @.str, double noundef %53) #21
  %call2.i1194 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1192) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1192) #21
  br label %if.end327

if.end327:                                        ; preds = %if.else325, %if.then322
  %strlen987 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr988 = getelementptr i8, ptr %call, i64 %strlen987
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(5) %endptr988, ptr noundef nonnull align 1 dereferenceable(5) @.str.20, i64 5, i1 false)
  %54 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char0989 = load i8, ptr %54, align 1
  %cmp331.not = icmp eq i8 %char0989, 0
  br i1 %cmp331.not, label %if.else335, label %if.then332

if.then332:                                       ; preds = %if.end327
  %call334 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %54) #21
  br label %if.end337

if.else335:                                       ; preds = %if.end327
  %55 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1195) #21
  %call.i1196 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1195, i64 noundef 15, ptr noundef nonnull @.str, double noundef %55) #21
  %call2.i1197 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1195) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1195) #21
  br label %if.end337

if.end337:                                        ; preds = %if.else335, %if.then332
  %strlen990 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr991 = getelementptr i8, ptr %call, i64 %strlen990
  store i16 42, ptr %endptr991, align 1
  %56 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char0992 = load i8, ptr %56, align 1
  %cmp341.not = icmp eq i8 %char0992, 0
  br i1 %cmp341.not, label %if.else345, label %if.then342

if.then342:                                       ; preds = %if.end337
  %call344 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %56) #21
  br label %if.end347

if.else345:                                       ; preds = %if.end337
  %57 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1198) #21
  %call.i1199 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1198, i64 noundef 15, ptr noundef nonnull @.str, double noundef %57) #21
  %call2.i1200 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1198) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1198) #21
  br label %if.end347

if.end347:                                        ; preds = %if.else345, %if.then342
  %strlen993 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr994 = getelementptr i8, ptr %call, i64 %strlen993
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(13) %endptr994, ptr noundef nonnull align 1 dereferenceable(13) @.str.21, i64 13, i1 false)
  %58 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char0995 = load i8, ptr %58, align 1
  %cmp351.not = icmp eq i8 %char0995, 0
  br i1 %cmp351.not, label %if.else355, label %if.then352

if.then352:                                       ; preds = %if.end347
  %call354 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %58) #21
  br label %if.end357

if.else355:                                       ; preds = %if.end347
  %59 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1201) #21
  %call.i1202 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1201, i64 noundef 15, ptr noundef nonnull @.str, double noundef %59) #21
  %call2.i1203 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1201) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1201) #21
  br label %if.end357

if.end357:                                        ; preds = %if.else355, %if.then352
  %strlen996 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr997 = getelementptr i8, ptr %call, i64 %strlen996
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(3) %endptr997, ptr noundef nonnull align 1 dereferenceable(3) @.str.15, i64 3, i1 false)
  %strlen998 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr999 = getelementptr i8, ptr %call, i64 %strlen998
  store i16 41, ptr %endptr999, align 1
  br label %sw.epilog

sw.bb360:                                         ; preds = %if.end
  %strlen967 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr968 = getelementptr i8, ptr %call, i64 %strlen967
  store i16 40, ptr %endptr968, align 1
  %60 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char0969 = load i8, ptr %60, align 1
  %cmp364.not = icmp eq i8 %char0969, 0
  br i1 %cmp364.not, label %if.else368, label %if.then365

if.then365:                                       ; preds = %sw.bb360
  %call367 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %60) #21
  br label %if.end370

if.else368:                                       ; preds = %sw.bb360
  %61 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1204) #21
  %call.i1205 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1204, i64 noundef 15, ptr noundef nonnull @.str, double noundef %61) #21
  %call2.i1206 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1204) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1204) #21
  br label %if.end370

if.end370:                                        ; preds = %if.else368, %if.then365
  %strlen970 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr971 = getelementptr i8, ptr %call, i64 %strlen970
  store i16 42, ptr %endptr971, align 1
  %strlen972 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr973 = getelementptr i8, ptr %call, i64 %strlen972
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(6) %endptr973, ptr noundef nonnull align 1 dereferenceable(6) @.str.22, i64 6, i1 false)
  %62 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char0974 = load i8, ptr %62, align 1
  %cmp375.not = icmp eq i8 %char0974, 0
  br i1 %cmp375.not, label %if.else379, label %if.then376

if.then376:                                       ; preds = %if.end370
  %call378 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %62) #21
  br label %if.end381

if.else379:                                       ; preds = %if.end370
  %63 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1207) #21
  %call.i1208 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1207, i64 noundef 15, ptr noundef nonnull @.str, double noundef %63) #21
  %call2.i1209 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1207) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1207) #21
  br label %if.end381

if.end381:                                        ; preds = %if.else379, %if.then376
  %strlen975 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr976 = getelementptr i8, ptr %call, i64 %strlen975
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(3) %endptr976, ptr noundef nonnull align 1 dereferenceable(3) @.str.10, i64 3, i1 false)
  %strlen977 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr978 = getelementptr i8, ptr %call, i64 %strlen977
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(6) %endptr978, ptr noundef nonnull align 1 dereferenceable(6) @.str.23, i64 6, i1 false)
  %64 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char0979 = load i8, ptr %64, align 1
  %cmp386.not = icmp eq i8 %char0979, 0
  br i1 %cmp386.not, label %if.else390, label %if.then387

if.then387:                                       ; preds = %if.end381
  %call389 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %64) #21
  br label %if.end392

if.else390:                                       ; preds = %if.end381
  %65 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1210) #21
  %call.i1211 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1210, i64 noundef 15, ptr noundef nonnull @.str, double noundef %65) #21
  %call2.i1212 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1210) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1210) #21
  br label %if.end392

if.end392:                                        ; preds = %if.else390, %if.then387
  %strlen980 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr981 = getelementptr i8, ptr %call, i64 %strlen980
  store i16 41, ptr %endptr981, align 1
  %strlen982 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr983 = getelementptr i8, ptr %call, i64 %strlen982
  store i16 41, ptr %endptr983, align 1
  br label %sw.epilog

sw.bb395:                                         ; preds = %if.end
  %strlen955 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr956 = getelementptr i8, ptr %call, i64 %strlen955
  store i16 40, ptr %endptr956, align 1
  %66 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char0957 = load i8, ptr %66, align 1
  %cmp399.not = icmp eq i8 %char0957, 0
  br i1 %cmp399.not, label %if.else403, label %if.then400

if.then400:                                       ; preds = %sw.bb395
  %call402 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %66) #21
  br label %if.end405

if.else403:                                       ; preds = %sw.bb395
  %67 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1213) #21
  %call.i1214 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1213, i64 noundef 15, ptr noundef nonnull @.str, double noundef %67) #21
  %call2.i1215 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1213) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1213) #21
  br label %if.end405

if.end405:                                        ; preds = %if.else403, %if.then400
  %strlen958 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr959 = getelementptr i8, ptr %call, i64 %strlen958
  store i16 42, ptr %endptr959, align 1
  %strlen960 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr961 = getelementptr i8, ptr %call, i64 %strlen960
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(6) %endptr961, ptr noundef nonnull align 1 dereferenceable(6) @.str.24, i64 6, i1 false)
  %68 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char0962 = load i8, ptr %68, align 1
  %cmp410.not = icmp eq i8 %char0962, 0
  br i1 %cmp410.not, label %if.else414, label %if.then411

if.then411:                                       ; preds = %if.end405
  %call413 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %68) #21
  br label %if.end416

if.else414:                                       ; preds = %if.end405
  %69 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1216) #21
  %call.i1217 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1216, i64 noundef 15, ptr noundef nonnull @.str, double noundef %69) #21
  %call2.i1218 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1216) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1216) #21
  br label %if.end416

if.end416:                                        ; preds = %if.else414, %if.then411
  %strlen963 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr964 = getelementptr i8, ptr %call, i64 %strlen963
  store i16 41, ptr %endptr964, align 1
  %strlen965 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr966 = getelementptr i8, ptr %call, i64 %strlen965
  store i16 41, ptr %endptr966, align 1
  br label %sw.epilog

sw.bb419:                                         ; preds = %if.end
  %strlen938 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr939 = getelementptr i8, ptr %call, i64 %strlen938
  store i16 40, ptr %endptr939, align 1
  %70 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char0940 = load i8, ptr %70, align 1
  %cmp423.not = icmp eq i8 %char0940, 0
  br i1 %cmp423.not, label %if.else427, label %if.then424

if.then424:                                       ; preds = %sw.bb419
  %call426 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %70) #21
  br label %if.end429

if.else427:                                       ; preds = %sw.bb419
  %71 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1219) #21
  %call.i1220 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1219, i64 noundef 15, ptr noundef nonnull @.str, double noundef %71) #21
  %call2.i1221 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1219) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1219) #21
  br label %if.end429

if.end429:                                        ; preds = %if.else427, %if.then424
  %strlen941 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr942 = getelementptr i8, ptr %call, i64 %strlen941
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(3) %endptr942, ptr noundef nonnull align 1 dereferenceable(3) @.str.13, i64 3, i1 false)
  %strlen943 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr944 = getelementptr i8, ptr %call, i64 %strlen943
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(6) %endptr944, ptr noundef nonnull align 1 dereferenceable(6) @.str.23, i64 6, i1 false)
  %72 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char0945 = load i8, ptr %72, align 1
  %cmp434.not = icmp eq i8 %char0945, 0
  br i1 %cmp434.not, label %if.else438, label %if.then435

if.then435:                                       ; preds = %if.end429
  %call437 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %72) #21
  br label %if.end440

if.else438:                                       ; preds = %if.end429
  %73 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1222) #21
  %call.i1223 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1222, i64 noundef 15, ptr noundef nonnull @.str, double noundef %73) #21
  %call2.i1224 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1222) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1222) #21
  br label %if.end440

if.end440:                                        ; preds = %if.else438, %if.then435
  %strlen946 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr947 = getelementptr i8, ptr %call, i64 %strlen946
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(3) %endptr947, ptr noundef nonnull align 1 dereferenceable(3) @.str.14, i64 3, i1 false)
  %strlen948 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr949 = getelementptr i8, ptr %call, i64 %strlen948
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(6) %endptr949, ptr noundef nonnull align 1 dereferenceable(6) @.str.22, i64 6, i1 false)
  %74 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char0950 = load i8, ptr %74, align 1
  %cmp445.not = icmp eq i8 %char0950, 0
  br i1 %cmp445.not, label %if.else449, label %if.then446

if.then446:                                       ; preds = %if.end440
  %call448 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %74) #21
  br label %if.end451

if.else449:                                       ; preds = %if.end440
  %75 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1225) #21
  %call.i1226 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1225, i64 noundef 15, ptr noundef nonnull @.str, double noundef %75) #21
  %call2.i1227 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1225) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1225) #21
  br label %if.end451

if.end451:                                        ; preds = %if.else449, %if.then446
  %strlen951 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr952 = getelementptr i8, ptr %call, i64 %strlen951
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(3) %endptr952, ptr noundef nonnull align 1 dereferenceable(3) @.str.15, i64 3, i1 false)
  %strlen953 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr954 = getelementptr i8, ptr %call, i64 %strlen953
  store i16 41, ptr %endptr954, align 1
  br label %sw.epilog

sw.bb454:                                         ; preds = %if.end
  %strlen933 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr934 = getelementptr i8, ptr %call, i64 %strlen933
  store i16 40, ptr %endptr934, align 1
  %76 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char0935 = load i8, ptr %76, align 1
  %cmp458.not = icmp eq i8 %char0935, 0
  br i1 %cmp458.not, label %if.else462, label %if.then459

if.then459:                                       ; preds = %sw.bb454
  %call461 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %76) #21
  br label %if.end464

if.else462:                                       ; preds = %sw.bb454
  %77 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1228) #21
  %call.i1229 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1228, i64 noundef 15, ptr noundef nonnull @.str, double noundef %77) #21
  %call2.i1230 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1228) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1228) #21
  br label %if.end464

if.end464:                                        ; preds = %if.else462, %if.then459
  %strlen936 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr937 = getelementptr i8, ptr %call, i64 %strlen936
  store i16 41, ptr %endptr937, align 1
  br label %sw.epilog

sw.bb466:                                         ; preds = %if.end
  %strlen926 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr927 = getelementptr i8, ptr %call, i64 %strlen926
  store i16 40, ptr %endptr927, align 1
  %strlen928 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr929 = getelementptr i8, ptr %call, i64 %strlen928
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %endptr929, ptr noundef nonnull align 1 dereferenceable(7) @.str.25, i64 7, i1 false)
  %78 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char0930 = load i8, ptr %78, align 1
  %cmp471.not = icmp eq i8 %char0930, 0
  br i1 %cmp471.not, label %if.else475, label %if.then472

if.then472:                                       ; preds = %sw.bb466
  %call474 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %78) #21
  br label %if.end477

if.else475:                                       ; preds = %sw.bb466
  %79 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1231) #21
  %call.i1232 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1231, i64 noundef 15, ptr noundef nonnull @.str, double noundef %79) #21
  %call2.i1233 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1231) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1231) #21
  br label %if.end477

if.end477:                                        ; preds = %if.else475, %if.then472
  %strlen931 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr932 = getelementptr i8, ptr %call, i64 %strlen931
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(3) %endptr932, ptr noundef nonnull align 1 dereferenceable(3) @.str.15, i64 3, i1 false)
  br label %sw.epilog

sw.bb479:                                         ; preds = %if.end
  %strlen924 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr925 = getelementptr i8, ptr %call, i64 %strlen924
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(6) %endptr925, ptr noundef nonnull align 1 dereferenceable(6) @.str.26, i64 6, i1 false)
  br label %sw.epilog

sw.bb481:                                         ; preds = %if.end
  %strlen922 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr923 = getelementptr i8, ptr %call, i64 %strlen922
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(6) %endptr923, ptr noundef nonnull align 1 dereferenceable(6) @.str.7, i64 6, i1 false)
  br label %sw.epilog

sw.bb483:                                         ; preds = %if.end
  %80 = icmp ult i32 %WRT, 2
  br i1 %80, label %if.then486, label %if.else509

if.then486:                                       ; preds = %sw.bb483
  %strlen912 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr913 = getelementptr i8, ptr %call, i64 %strlen912
  store i16 40, ptr %endptr913, align 1
  %strlen914 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr915 = getelementptr i8, ptr %call, i64 %strlen914
  store i16 40, ptr %endptr915, align 1
  %81 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char0916 = load i8, ptr %81, align 1
  %cmp491.not = icmp eq i8 %char0916, 0
  br i1 %cmp491.not, label %if.else495, label %if.then492

if.then492:                                       ; preds = %if.then486
  %call494 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %81) #21
  br label %if.end497

if.else495:                                       ; preds = %if.then486
  %82 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @fACAppendDoubleToString(ptr noundef %call, double noundef %82)
  br label %if.end497

if.end497:                                        ; preds = %if.else495, %if.then492
  %strlen917 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr918 = getelementptr i8, ptr %call, i64 %strlen917
  store i16 42, ptr %endptr918, align 1
  %arrayidx499 = getelementptr inbounds ptr, ptr %OperandNames, i64 1
  %83 = load ptr, ptr %arrayidx499, align 8, !tbaa !8
  %char0919 = load i8, ptr %83, align 1
  %cmp501.not = icmp eq i8 %char0919, 0
  br i1 %cmp501.not, label %if.else505, label %if.then502

if.then502:                                       ; preds = %if.end497
  %call504 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %83) #21
  br label %if.end507

if.else505:                                       ; preds = %if.end497
  %arrayidx506 = getelementptr inbounds double, ptr %OperandValues, i64 1
  %84 = load double, ptr %arrayidx506, align 8, !tbaa !10
  call void @fACAppendDoubleToString(ptr noundef %call, double noundef %84)
  br label %if.end507

if.end507:                                        ; preds = %if.else505, %if.then502
  %strlen920 = call i64 @strlen(ptr nonnull dereferenceable(1) %call)
  %endptr921 = getelementptr i8, ptr %call, i64 %strlen920
  store i16 41, ptr %endptr921, align 1
  br label %if.end523

if.else509:                                       ; preds = %sw.bb483
  %cmp510 = icmp eq i32 %WRT, 2
  br i1 %cmp510, label %if.then511, label %if.end523

if.then511:                                       ; preds = %if.else509
  %arrayidx512 = getelementptr inbounds ptr, ptr %OperandNames, i64 2
  %85 = load ptr, ptr %arrayidx512, align 8, !tbaa !8
  %char0909 = load i8, ptr %85, align 1
  %cmp514.not = icmp eq i8 %char0909, 0
  br i1 %cmp514.not, label %if.else519, label %if.then515

if.then515:                                       ; preds = %if.then511
  %strlen910 = call i64 @strlen(ptr nonnull %call)
  %endptr911 = getelementptr i8, ptr %call, i64 %strlen910
  store i16 40, ptr %endptr911, align 1
  %call518 = call ptr @strcat(ptr noundef nonnull %call, ptr noundef nonnull dereferenceable(1) %85) #21
  br label %if.end523

if.else519:                                       ; preds = %if.then511
  %arrayidx520 = getelementptr inbounds double, ptr %OperandValues, i64 2
  %86 = load double, ptr %arrayidx520, align 8, !tbaa !10
  call void @fACAppendDoubleToString(ptr noundef nonnull %call, double noundef %86)
  br label %if.end523

if.end523:                                        ; preds = %if.else509, %if.else519, %if.then515, %if.end507
  %strlen = call i64 @strlen(ptr nonnull %call)
  %endptr = getelementptr i8, ptr %call, i64 %strlen
  store i16 47, ptr %endptr, align 1
  %strlen893 = call i64 @strlen(ptr nonnull %call)
  %endptr894 = getelementptr i8, ptr %call, i64 %strlen893
  store i16 40, ptr %endptr894, align 1
  %strlen895 = call i64 @strlen(ptr nonnull %call)
  %endptr896 = getelementptr i8, ptr %call, i64 %strlen895
  store i16 40, ptr %endptr896, align 1
  %87 = load ptr, ptr %OperandNames, align 8, !tbaa !8
  %char0 = load i8, ptr %87, align 1
  %cmp529.not = icmp eq i8 %char0, 0
  br i1 %cmp529.not, label %if.else533, label %if.then530

if.then530:                                       ; preds = %if.end523
  %call532 = call ptr @strcat(ptr noundef nonnull %call, ptr noundef nonnull %87) #21
  br label %if.end535

if.else533:                                       ; preds = %if.end523
  %88 = load double, ptr %OperandValues, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1234) #21
  %call.i1235 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1234, i64 noundef 15, ptr noundef nonnull @.str, double noundef %88) #21
  %call2.i1236 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1234) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1234) #21
  br label %if.end535

if.end535:                                        ; preds = %if.else533, %if.then530
  %strlen897 = call i64 @strlen(ptr nonnull %call)
  %endptr898 = getelementptr i8, ptr %call, i64 %strlen897
  store i16 42, ptr %endptr898, align 1
  %arrayidx537 = getelementptr inbounds ptr, ptr %OperandNames, i64 1
  %89 = load ptr, ptr %arrayidx537, align 8, !tbaa !8
  %char0899 = load i8, ptr %89, align 1
  %cmp539.not = icmp eq i8 %char0899, 0
  br i1 %cmp539.not, label %if.else543, label %if.then540

if.then540:                                       ; preds = %if.end535
  %call542 = call ptr @strcat(ptr noundef nonnull %call, ptr noundef nonnull %89) #21
  br label %if.end545

if.else543:                                       ; preds = %if.end535
  %arrayidx544 = getelementptr inbounds double, ptr %OperandValues, i64 1
  %90 = load double, ptr %arrayidx544, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1237) #21
  %call.i1238 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1237, i64 noundef 15, ptr noundef nonnull @.str, double noundef %90) #21
  %call2.i1239 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1237) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1237) #21
  br label %if.end545

if.end545:                                        ; preds = %if.else543, %if.then540
  %strlen900 = call i64 @strlen(ptr nonnull %call)
  %endptr901 = getelementptr i8, ptr %call, i64 %strlen900
  store i16 41, ptr %endptr901, align 1
  %strlen902 = call i64 @strlen(ptr nonnull %call)
  %endptr903 = getelementptr i8, ptr %call, i64 %strlen902
  store i16 43, ptr %endptr903, align 1
  %arrayidx548 = getelementptr inbounds ptr, ptr %OperandNames, i64 2
  %91 = load ptr, ptr %arrayidx548, align 8, !tbaa !8
  %char0904 = load i8, ptr %91, align 1
  %cmp550.not = icmp eq i8 %char0904, 0
  br i1 %cmp550.not, label %if.else554, label %if.then551

if.then551:                                       ; preds = %if.end545
  %call553 = call ptr @strcat(ptr noundef nonnull %call, ptr noundef nonnull %91) #21
  br label %if.end556

if.else554:                                       ; preds = %if.end545
  %arrayidx555 = getelementptr inbounds double, ptr %OperandValues, i64 2
  %92 = load double, ptr %arrayidx555, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 15, ptr nonnull %DoubleBuffer.i1240) #21
  %call.i1241 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr noundef nonnull %DoubleBuffer.i1240, i64 noundef 15, ptr noundef nonnull @.str, double noundef %92) #21
  %call2.i1242 = call ptr @strcat(ptr noundef nonnull dereferenceable(1) %call, ptr noundef nonnull %DoubleBuffer.i1240) #21
  call void @llvm.lifetime.end.p0(i64 15, ptr nonnull %DoubleBuffer.i1240) #21
  br label %if.end556

if.end556:                                        ; preds = %if.else554, %if.then551
  %strlen905 = call i64 @strlen(ptr nonnull %call)
  %endptr906 = getelementptr i8, ptr %call, i64 %strlen905
  store i16 41, ptr %endptr906, align 1
  %strlen907 = call i64 @strlen(ptr nonnull %call)
  %endptr908 = getelementptr i8, ptr %call, i64 %strlen907
  store i16 41, ptr %endptr908, align 1
  br label %sw.epilog

sw.default:                                       ; preds = %if.end
  %call559 = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.27, i32 noundef %F)
  call void @exit(i32 noundef 1) #23
  unreachable

sw.epilog:                                        ; preds = %if.end556, %sw.bb481, %sw.bb479, %if.end477, %if.end464, %if.end451, %if.end416, %if.end392, %if.end357, %if.end314, %if.end270, %if.end226, %if.end191, %if.end167, %sw.bb133, %if.end130, %if.end47
  ret ptr %call
}

; Function Attrs: inaccessiblememonly mustprogress nofree nounwind willreturn allocsize(0)
declare noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #6

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr nocapture noundef readonly, ...) local_unnamed_addr #3

; Function Attrs: noreturn nounwind
declare void @exit(i32 noundef) local_unnamed_addr #7

; Function Attrs: argmemonly mustprogress nofree nounwind readonly willreturn
declare i64 @strlen(ptr nocapture noundef) local_unnamed_addr #8

; Function Attrs: nofree nounwind uwtable
define dso_local void @fAFcreateLogDirectory(ptr nocapture noundef readonly %DirectoryName) local_unnamed_addr #1 {
entry:
  %ST = alloca %struct.stat, align 8
  call void @llvm.lifetime.start.p0(i64 144, ptr nonnull %ST) #21
  %call = call i32 @stat(ptr noundef %DirectoryName, ptr noundef nonnull %ST) #21
  %cmp = icmp eq i32 %call, -1
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call1 = call i32 @mkdir(ptr noundef %DirectoryName, i32 noundef 509) #21
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call void @llvm.lifetime.end.p0(i64 144, ptr nonnull %ST) #21
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @stat(ptr nocapture noundef readonly, ptr nocapture noundef) local_unnamed_addr #3

; Function Attrs: nofree nounwind
declare noundef i32 @mkdir(ptr nocapture noundef readonly, i32 noundef) local_unnamed_addr #3

; Function Attrs: nounwind uwtable
define dso_local void @fACGenerateExecutionID(ptr noundef %ExecutionId) local_unnamed_addr #5 {
entry:
  %PIDStr = alloca [11 x i8], align 1
  store i8 0, ptr %ExecutionId, align 1, !tbaa !5
  %call = call i32 @gethostname(ptr noundef nonnull %ExecutionId, i64 noundef 256) #21
  %cmp.not = icmp eq i32 %call, 0
  br i1 %cmp.not, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(13) %ExecutionId, ptr noundef nonnull align 1 dereferenceable(13) @.str.28, i64 13, i1 false) #21
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %call2 = call i32 @getpid() #21
  call void @llvm.lifetime.start.p0(i64 11, ptr nonnull %PIDStr) #21
  store i8 0, ptr %PIDStr, align 1, !tbaa !5
  %call4 = call i32 (ptr, ptr, ...) @sprintf(ptr noundef nonnull %PIDStr, ptr noundef nonnull @.str.29, i32 noundef %call2) #21
  %strlen = call i64 @strlen(ptr nonnull %ExecutionId)
  %endptr = getelementptr i8, ptr %ExecutionId, i64 %strlen
  store i16 95, ptr %endptr, align 1
  %call7 = call ptr @strcat(ptr noundef nonnull %ExecutionId, ptr noundef nonnull %PIDStr) #21
  call void @llvm.lifetime.end.p0(i64 11, ptr nonnull %PIDStr) #21
  ret void
}

; Function Attrs: nounwind
declare i32 @gethostname(ptr noundef, i64 noundef) local_unnamed_addr #9

; Function Attrs: argmemonly mustprogress nofree nounwind willreturn
declare ptr @strcpy(ptr noalias noundef returned writeonly, ptr noalias nocapture noundef readonly) local_unnamed_addr #4

; Function Attrs: nounwind
declare i32 @getpid() local_unnamed_addr #9

; Function Attrs: nofree nounwind
declare noundef i32 @sprintf(ptr noalias nocapture noundef writeonly, ptr nocapture noundef readonly, ...) local_unnamed_addr #3

; Function Attrs: nounwind uwtable
define dso_local void @fAFGenerateFileString(ptr noundef %File, ptr nocapture noundef readonly %FileNamePrefix, ptr nocapture noundef readonly %Extension) local_unnamed_addr #5 {
entry:
  %PIDStr.i = alloca [11 x i8], align 1
  %ST.i = alloca %struct.stat, align 8
  %ExecutionId = alloca [5000 x i8], align 16
  %cmp.not = icmp eq ptr %File, null
  br i1 %cmp.not, label %if.else, label %if.end

if.else:                                          ; preds = %entry
  call void @__assert_fail(ptr noundef nonnull @.str.32, ptr noundef nonnull @.str.33, i32 noundef 577, ptr noundef nonnull @__PRETTY_FUNCTION__.fAFGenerateFileString) #23
  unreachable

if.end:                                           ; preds = %entry
  %vla9 = alloca [10 x i8], align 16
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 16 dereferenceable(10) %vla9, ptr noundef nonnull align 1 dereferenceable(10) @.str.34, i64 10, i1 false) #21
  call void @llvm.lifetime.start.p0(i64 144, ptr nonnull %ST.i) #21
  %call.i = call i32 @stat(ptr noundef nonnull %vla9, ptr noundef nonnull %ST.i) #21
  %cmp.i = icmp eq i32 %call.i, -1
  br i1 %cmp.i, label %if.then.i, label %fAFcreateLogDirectory.exit

if.then.i:                                        ; preds = %if.end
  %call1.i = call i32 @mkdir(ptr noundef nonnull %vla9, i32 noundef 509) #21
  br label %fAFcreateLogDirectory.exit

fAFcreateLogDirectory.exit:                       ; preds = %if.end, %if.then.i
  call void @llvm.lifetime.end.p0(i64 144, ptr nonnull %ST.i) #21
  call void @llvm.lifetime.start.p0(i64 5000, ptr nonnull %ExecutionId) #21
  store i8 0, ptr %ExecutionId, align 16, !tbaa !5
  %call.i10 = call i32 @gethostname(ptr noundef nonnull %ExecutionId, i64 noundef 256) #21
  %cmp.not.i = icmp eq i32 %call.i10, 0
  br i1 %cmp.not.i, label %fACGenerateExecutionID.exit, label %if.then.i11

if.then.i11:                                      ; preds = %fAFcreateLogDirectory.exit
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 16 dereferenceable(13) %ExecutionId, ptr noundef nonnull align 1 dereferenceable(13) @.str.28, i64 13, i1 false) #21
  br label %fACGenerateExecutionID.exit

fACGenerateExecutionID.exit:                      ; preds = %fAFcreateLogDirectory.exit, %if.then.i11
  %call2.i = call i32 @getpid() #21
  call void @llvm.lifetime.start.p0(i64 11, ptr nonnull %PIDStr.i) #21
  store i8 0, ptr %PIDStr.i, align 1, !tbaa !5
  %call4.i = call i32 (ptr, ptr, ...) @sprintf(ptr noundef nonnull %PIDStr.i, ptr noundef nonnull @.str.29, i32 noundef %call2.i) #21
  %strlen.i = call i64 @strlen(ptr nonnull %ExecutionId) #21
  %endptr.i = getelementptr i8, ptr %ExecutionId, i64 %strlen.i
  store i16 95, ptr %endptr.i, align 1
  %call7.i = call ptr @strcat(ptr noundef nonnull %ExecutionId, ptr noundef nonnull %PIDStr.i) #21
  call void @llvm.lifetime.end.p0(i64 11, ptr nonnull %PIDStr.i) #21
  store i8 0, ptr %File, align 1, !tbaa !5
  %call1 = call ptr @strcpy(ptr noundef nonnull %File, ptr noundef nonnull %vla9) #21
  %strlen = call i64 @strlen(ptr nonnull %File)
  %endptr = getelementptr i8, ptr %File, i64 %strlen
  store i16 47, ptr %endptr, align 1
  %call3 = call ptr @strcat(ptr noundef nonnull %File, ptr noundef nonnull dereferenceable(1) %FileNamePrefix) #21
  %call5 = call ptr @strcat(ptr noundef nonnull %File, ptr noundef nonnull %ExecutionId) #21
  %call6 = call ptr @strcat(ptr noundef nonnull %File, ptr noundef nonnull dereferenceable(1) %Extension) #21
  call void @llvm.lifetime.end.p0(i64 5000, ptr nonnull %ExecutionId) #21
  ret void
}

; Function Attrs: noreturn nounwind
declare void @__assert_fail(ptr noundef, ptr noundef, i32 noundef, ptr noundef) local_unnamed_addr #7

; Function Attrs: nounwind uwtable
define dso_local void @fACCreate() local_unnamed_addr #5 {
entry:
  %call = call noalias dereferenceable_or_null(16) ptr @malloc(i64 noundef 16) #22
  store ptr %call, ptr @ACs, align 8, !tbaa !8
  %cmp = icmp eq ptr %call, null
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call1 = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.35)
  call void @exit(i32 noundef 1) #23
  unreachable

if.end:                                           ; preds = %entry
  %call2 = call noalias dereferenceable_or_null(8000000) ptr @malloc(i64 noundef 8000000) #22
  %ACItems = getelementptr inbounds %struct.ACTable, ptr %call, i64 0, i32 1
  store ptr %call2, ptr %ACItems, align 8, !tbaa !12
  %cmp3 = icmp eq ptr %call2, null
  br i1 %cmp3, label %if.then4, label %if.end6

if.then4:                                         ; preds = %if.end
  %call5 = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.36)
  call void @exit(i32 noundef 1) #23
  unreachable

if.end6:                                          ; preds = %if.end
  store i64 0, ptr %call, align 8, !tbaa !15
  store i32 0, ptr @ACItemCounter, align 4, !tbaa !16
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local void @fCreateACItem(ptr nocapture noundef writeonly %AddressToAllocateAt) local_unnamed_addr #5 {
entry:
  %call = call noalias dereferenceable_or_null(72) ptr @malloc(i64 noundef 72) #22
  store ptr %call, ptr %AddressToAllocateAt, align 8, !tbaa !8
  %cmp = icmp eq ptr %call, null
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call1 = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.37)
  call void @exit(i32 noundef 1) #23
  unreachable

if.end:                                           ; preds = %entry
  %0 = load i32, ptr @ACItemCounter, align 4, !tbaa !16
  %inc = add nsw i32 %0, 1
  store i32 %inc, ptr @ACItemCounter, align 4, !tbaa !16
  ret void
}

; Function Attrs: argmemonly mustprogress nofree norecurse nosync nounwind readonly willreturn uwtable
define dso_local i32 @fACItemsEqual(ptr nocapture noundef readonly %X, ptr nocapture noundef readonly %Y) local_unnamed_addr #10 {
entry:
  %0 = load i32, ptr %X, align 8, !tbaa !18
  %1 = load i32, ptr %Y, align 8, !tbaa !18
  %cmp = icmp eq i32 %0, %1
  %. = zext i1 %cmp to i32
  ret i32 %.
}

; Function Attrs: nounwind uwtable
define dso_local void @fACSetACItem(ptr noundef %AtomicConditionsTable, ptr noundef readonly %NewValue) local_unnamed_addr #5 {
entry:
  %cmp = icmp eq ptr %AtomicConditionsTable, null
  br i1 %cmp, label %return, label %if.end

if.end:                                           ; preds = %entry
  %ACItems = getelementptr inbounds %struct.ACTable, ptr %AtomicConditionsTable, i64 0, i32 1
  %0 = load ptr, ptr %ACItems, align 8, !tbaa !12
  %1 = load i32, ptr %NewValue, align 8, !tbaa !18
  %idxprom = sext i32 %1 to i64
  %arrayidx = getelementptr inbounds ptr, ptr %0, i64 %idxprom
  %2 = load ptr, ptr %arrayidx, align 8, !tbaa !8
  %cmp1.not = icmp eq ptr %2, null
  br i1 %cmp1.not, label %if.else, label %land.lhs.true

land.lhs.true:                                    ; preds = %if.end
  %3 = load i32, ptr %2, align 8, !tbaa !18
  %cmp.i.not = icmp eq i32 %1, %3
  br i1 %cmp.i.not, label %if.then2, label %if.else

if.then2:                                         ; preds = %land.lhs.true
  %F = getelementptr inbounds %struct.ACItem, ptr %NewValue, i64 0, i32 1
  %4 = load i32, ptr %F, align 4, !tbaa !20
  %F3 = getelementptr inbounds %struct.ACItem, ptr %2, i64 0, i32 1
  store i32 %4, ptr %F3, align 4, !tbaa !20
  %5 = add i32 %4, -17
  %switch.i.i = icmp ult i32 %5, -13
  br i1 %switch.i.i, label %if.end.i, label %fACFuncHasXNumOperands.exit

if.end.i:                                         ; preds = %if.then2
  %6 = icmp ugt i32 %4, 3
  br i1 %6, label %if.end4.i, label %fACFuncHasXNumOperands.exit

if.end4.i:                                        ; preds = %if.end.i
  %cmp.i.not.i = icmp eq i32 %4, 17
  %..i230 = select i1 %cmp.i.not.i, i32 3, i32 0
  br label %fACFuncHasXNumOperands.exit

fACFuncHasXNumOperands.exit:                      ; preds = %if.then2, %if.end.i, %if.end4.i
  %retval.0.i = phi i32 [ 1, %if.then2 ], [ 2, %if.end.i ], [ %..i230, %if.end4.i ]
  %NumOperands = getelementptr inbounds %struct.ACItem, ptr %2, i64 0, i32 2
  store i32 %retval.0.i, ptr %NumOperands, align 8, !tbaa !21
  %ResultVar = getelementptr inbounds %struct.ACItem, ptr %NewValue, i64 0, i32 3
  %7 = load ptr, ptr %ResultVar, align 8, !tbaa !22
  %ResultVar6 = getelementptr inbounds %struct.ACItem, ptr %2, i64 0, i32 3
  store ptr %7, ptr %ResultVar6, align 8, !tbaa !22
  %cmp8240.not = icmp eq i32 %retval.0.i, 0
  br i1 %cmp8240.not, label %for.cond.cleanup, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %fACFuncHasXNumOperands.exit
  %OperandNames = getelementptr inbounds %struct.ACItem, ptr %NewValue, i64 0, i32 4
  %OperandNames11 = getelementptr inbounds %struct.ACItem, ptr %2, i64 0, i32 4
  %OperandValues = getelementptr inbounds %struct.ACItem, ptr %NewValue, i64 0, i32 5
  %OperandValues16 = getelementptr inbounds %struct.ACItem, ptr %2, i64 0, i32 5
  %ACStrings = getelementptr inbounds %struct.ACItem, ptr %2, i64 0, i32 7
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %fACFuncHasXNumOperands.exit
  %ACWRTOperands = getelementptr inbounds %struct.ACItem, ptr %NewValue, i64 0, i32 6
  %8 = load ptr, ptr %ACWRTOperands, align 8, !tbaa !23
  %ACWRTOperands25 = getelementptr inbounds %struct.ACItem, ptr %2, i64 0, i32 6
  store ptr %8, ptr %ACWRTOperands25, align 8, !tbaa !23
  %FileName = getelementptr inbounds %struct.ACItem, ptr %NewValue, i64 0, i32 8
  %9 = load ptr, ptr %FileName, align 8, !tbaa !24
  %FileName26 = getelementptr inbounds %struct.ACItem, ptr %2, i64 0, i32 8
  store ptr %9, ptr %FileName26, align 8, !tbaa !24
  %LineNumber = getelementptr inbounds %struct.ACItem, ptr %NewValue, i64 0, i32 9
  %10 = load i32, ptr %LineNumber, align 8, !tbaa !25
  %LineNumber27 = getelementptr inbounds %struct.ACItem, ptr %2, i64 0, i32 9
  store i32 %10, ptr %LineNumber27, align 8, !tbaa !25
  br label %return

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %11 = load ptr, ptr %OperandNames, align 8, !tbaa !26
  %arrayidx10 = getelementptr inbounds ptr, ptr %11, i64 %indvars.iv
  %12 = load ptr, ptr %arrayidx10, align 8, !tbaa !8
  %13 = load ptr, ptr %OperandNames11, align 8, !tbaa !26
  %arrayidx13 = getelementptr inbounds ptr, ptr %13, i64 %indvars.iv
  store ptr %12, ptr %arrayidx13, align 8, !tbaa !8
  %14 = load ptr, ptr %OperandValues, align 8, !tbaa !27
  %arrayidx15 = getelementptr inbounds double, ptr %14, i64 %indvars.iv
  %15 = load double, ptr %arrayidx15, align 8, !tbaa !10
  %16 = load ptr, ptr %OperandValues16, align 8, !tbaa !27
  %arrayidx18 = getelementptr inbounds double, ptr %16, i64 %indvars.iv
  store double %15, ptr %arrayidx18, align 8, !tbaa !10
  %17 = load ptr, ptr %OperandNames, align 8, !tbaa !26
  %18 = load i32, ptr %F, align 4, !tbaa !20
  %19 = trunc i64 %indvars.iv to i32
  %call22 = call ptr @fACDumpAtomicConditionString(ptr noundef %17, ptr noundef %14, i32 noundef %18, i32 noundef %19)
  %20 = load ptr, ptr %ACStrings, align 8, !tbaa !28
  %arrayidx24 = getelementptr inbounds ptr, ptr %20, i64 %indvars.iv
  store ptr %call22, ptr %arrayidx24, align 8, !tbaa !8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %21 = load i32, ptr %NumOperands, align 8, !tbaa !21
  %22 = sext i32 %21 to i64
  %cmp8 = icmp slt i64 %indvars.iv.next, %22
  br i1 %cmp8, label %for.body, label %for.cond.cleanup, !llvm.loop !29

if.else:                                          ; preds = %land.lhs.true, %if.end
  %call.i = call noalias dereferenceable_or_null(72) ptr @malloc(i64 noundef 72) #22
  store ptr %call.i, ptr %arrayidx, align 8, !tbaa !8
  %cmp.i231 = icmp eq ptr %call.i, null
  br i1 %cmp.i231, label %if.then.i, label %fCreateACItem.exit

if.then.i:                                        ; preds = %if.else
  %call1.i = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.37) #21
  call void @exit(i32 noundef 1) #23
  unreachable

fCreateACItem.exit:                               ; preds = %if.else
  %23 = load i32, ptr @ACItemCounter, align 4, !tbaa !16
  %inc.i = add nsw i32 %23, 1
  store i32 %inc.i, ptr @ACItemCounter, align 4, !tbaa !16
  %24 = load i32, ptr %NewValue, align 8, !tbaa !18
  %25 = load ptr, ptr %ACItems, align 8, !tbaa !12
  %idxprom35 = sext i32 %24 to i64
  %arrayidx36 = getelementptr inbounds ptr, ptr %25, i64 %idxprom35
  %26 = load ptr, ptr %arrayidx36, align 8, !tbaa !8
  store i32 %24, ptr %26, align 8, !tbaa !18
  %F38 = getelementptr inbounds %struct.ACItem, ptr %NewValue, i64 0, i32 1
  %27 = load i32, ptr %F38, align 4, !tbaa !20
  %28 = load i32, ptr %NewValue, align 8, !tbaa !18
  %idxprom41 = sext i32 %28 to i64
  %arrayidx42 = getelementptr inbounds ptr, ptr %25, i64 %idxprom41
  %29 = load ptr, ptr %arrayidx42, align 8, !tbaa !8
  %F43 = getelementptr inbounds %struct.ACItem, ptr %29, i64 0, i32 1
  store i32 %27, ptr %F43, align 4, !tbaa !20
  %30 = load i32, ptr %F38, align 4, !tbaa !20
  %31 = add i32 %30, -17
  %switch.i.i233 = icmp ult i32 %31, -13
  br i1 %switch.i.i233, label %if.end.i234, label %fACFuncHasXNumOperands.exit239

if.end.i234:                                      ; preds = %fCreateACItem.exit
  %32 = icmp ugt i32 %30, 3
  br i1 %32, label %if.end4.i237, label %fACFuncHasXNumOperands.exit239

if.end4.i237:                                     ; preds = %if.end.i234
  %cmp.i.not.i235 = icmp eq i32 %30, 17
  %..i236 = select i1 %cmp.i.not.i235, i32 3, i32 0
  br label %fACFuncHasXNumOperands.exit239

fACFuncHasXNumOperands.exit239:                   ; preds = %fCreateACItem.exit, %if.end.i234, %if.end4.i237
  %retval.0.i238 = phi i32 [ 1, %fCreateACItem.exit ], [ 2, %if.end.i234 ], [ %..i236, %if.end4.i237 ]
  %NumOperands50 = getelementptr inbounds %struct.ACItem, ptr %29, i64 0, i32 2
  store i32 %retval.0.i238, ptr %NumOperands50, align 8, !tbaa !21
  %ResultVar51 = getelementptr inbounds %struct.ACItem, ptr %NewValue, i64 0, i32 3
  %33 = load ptr, ptr %ResultVar51, align 8, !tbaa !22
  %ResultVar56 = getelementptr inbounds %struct.ACItem, ptr %29, i64 0, i32 3
  store ptr %33, ptr %ResultVar56, align 8, !tbaa !22
  %34 = load ptr, ptr %arrayidx42, align 8, !tbaa !8
  %NumOperands61 = getelementptr inbounds %struct.ACItem, ptr %34, i64 0, i32 2
  %35 = load i32, ptr %NumOperands61, align 8, !tbaa !21
  %conv = sext i32 %35 to i64
  %mul = shl nsw i64 %conv, 3
  %call62 = call noalias ptr @malloc(i64 noundef %mul) #22
  %OperandNames67 = getelementptr inbounds %struct.ACItem, ptr %34, i64 0, i32 4
  store ptr %call62, ptr %OperandNames67, align 8, !tbaa !26
  %cmp68 = icmp eq ptr %call62, null
  br i1 %cmp68, label %if.then70, label %if.end72

if.then70:                                        ; preds = %fACFuncHasXNumOperands.exit239
  %call71 = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.38)
  call void @exit(i32 noundef 1) #23
  unreachable

if.end72:                                         ; preds = %fACFuncHasXNumOperands.exit239
  %36 = load ptr, ptr %arrayidx42, align 8, !tbaa !8
  %NumOperands77 = getelementptr inbounds %struct.ACItem, ptr %36, i64 0, i32 2
  %37 = load i32, ptr %NumOperands77, align 8, !tbaa !21
  %conv78 = sext i32 %37 to i64
  %mul79 = shl nsw i64 %conv78, 3
  %call80 = call noalias ptr @malloc(i64 noundef %mul79) #22
  %OperandValues85 = getelementptr inbounds %struct.ACItem, ptr %36, i64 0, i32 5
  store ptr %call80, ptr %OperandValues85, align 8, !tbaa !27
  %cmp86 = icmp eq ptr %call80, null
  br i1 %cmp86, label %if.then88, label %for.cond92.preheader

for.cond92.preheader:                             ; preds = %if.end72
  %38 = load ptr, ptr %ACItems, align 8, !tbaa !12
  %arrayidx96243 = getelementptr inbounds ptr, ptr %38, i64 %idxprom41
  %39 = load ptr, ptr %arrayidx96243, align 8, !tbaa !8
  %NumOperands97244 = getelementptr inbounds %struct.ACItem, ptr %39, i64 0, i32 2
  %40 = load i32, ptr %NumOperands97244, align 8, !tbaa !21
  %cmp98245 = icmp sgt i32 %40, 0
  br i1 %cmp98245, label %for.body101.lr.ph, label %for.cond.cleanup100

for.body101.lr.ph:                                ; preds = %for.cond92.preheader
  %OperandNames102 = getelementptr inbounds %struct.ACItem, ptr %NewValue, i64 0, i32 4
  %OperandValues112 = getelementptr inbounds %struct.ACItem, ptr %NewValue, i64 0, i32 5
  br label %for.body101

if.then88:                                        ; preds = %if.end72
  %call89 = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.39)
  call void @exit(i32 noundef 1) #23
  unreachable

for.cond.cleanup100:                              ; preds = %for.body101, %for.cond92.preheader
  %.lcssa242 = phi ptr [ %38, %for.cond92.preheader ], [ %58, %for.body101 ]
  %.lcssa = phi ptr [ %39, %for.cond92.preheader ], [ %59, %for.body101 ]
  %arrayidx96.le = getelementptr inbounds ptr, ptr %.lcssa242, i64 %idxprom41
  %ACWRTOperands125 = getelementptr inbounds %struct.ACItem, ptr %NewValue, i64 0, i32 6
  %41 = load ptr, ptr %ACWRTOperands125, align 8, !tbaa !23
  %ACWRTOperands130 = getelementptr inbounds %struct.ACItem, ptr %.lcssa, i64 0, i32 6
  store ptr %41, ptr %ACWRTOperands130, align 8, !tbaa !23
  %ACStrings131 = getelementptr inbounds %struct.ACItem, ptr %NewValue, i64 0, i32 7
  %42 = load ptr, ptr %ACStrings131, align 8, !tbaa !28
  %43 = load ptr, ptr %arrayidx96.le, align 8, !tbaa !8
  %ACStrings136 = getelementptr inbounds %struct.ACItem, ptr %43, i64 0, i32 7
  store ptr %42, ptr %ACStrings136, align 8, !tbaa !28
  %FileName137 = getelementptr inbounds %struct.ACItem, ptr %NewValue, i64 0, i32 8
  %44 = load ptr, ptr %FileName137, align 8, !tbaa !24
  %45 = load ptr, ptr %arrayidx96.le, align 8, !tbaa !8
  %FileName142 = getelementptr inbounds %struct.ACItem, ptr %45, i64 0, i32 8
  store ptr %44, ptr %FileName142, align 8, !tbaa !24
  %LineNumber143 = getelementptr inbounds %struct.ACItem, ptr %NewValue, i64 0, i32 9
  %46 = load i32, ptr %LineNumber143, align 8, !tbaa !25
  %47 = load ptr, ptr %arrayidx96.le, align 8, !tbaa !8
  %LineNumber148 = getelementptr inbounds %struct.ACItem, ptr %47, i64 0, i32 9
  store i32 %46, ptr %LineNumber148, align 8, !tbaa !25
  %48 = load i64, ptr %AtomicConditionsTable, align 8, !tbaa !15
  %inc149 = add i64 %48, 1
  store i64 %inc149, ptr %AtomicConditionsTable, align 8, !tbaa !15
  br label %return

for.body101:                                      ; preds = %for.body101.lr.ph, %for.body101
  %indvars.iv251 = phi i64 [ 0, %for.body101.lr.ph ], [ %indvars.iv.next252, %for.body101 ]
  %49 = phi ptr [ %39, %for.body101.lr.ph ], [ %59, %for.body101 ]
  %50 = load ptr, ptr %OperandNames102, align 8, !tbaa !26
  %arrayidx104 = getelementptr inbounds ptr, ptr %50, i64 %indvars.iv251
  %51 = load ptr, ptr %arrayidx104, align 8, !tbaa !8
  %OperandNames109 = getelementptr inbounds %struct.ACItem, ptr %49, i64 0, i32 4
  %52 = load ptr, ptr %OperandNames109, align 8, !tbaa !26
  %arrayidx111 = getelementptr inbounds ptr, ptr %52, i64 %indvars.iv251
  store ptr %51, ptr %arrayidx111, align 8, !tbaa !8
  %53 = load ptr, ptr %OperandValues112, align 8, !tbaa !27
  %arrayidx114 = getelementptr inbounds double, ptr %53, i64 %indvars.iv251
  %54 = load double, ptr %arrayidx114, align 8, !tbaa !10
  %55 = load ptr, ptr %ACItems, align 8, !tbaa !12
  %arrayidx118 = getelementptr inbounds ptr, ptr %55, i64 %idxprom41
  %56 = load ptr, ptr %arrayidx118, align 8, !tbaa !8
  %OperandValues119 = getelementptr inbounds %struct.ACItem, ptr %56, i64 0, i32 5
  %57 = load ptr, ptr %OperandValues119, align 8, !tbaa !27
  %arrayidx121 = getelementptr inbounds double, ptr %57, i64 %indvars.iv251
  store double %54, ptr %arrayidx121, align 8, !tbaa !10
  %indvars.iv.next252 = add nuw nsw i64 %indvars.iv251, 1
  %58 = load ptr, ptr %ACItems, align 8, !tbaa !12
  %arrayidx96 = getelementptr inbounds ptr, ptr %58, i64 %idxprom41
  %59 = load ptr, ptr %arrayidx96, align 8, !tbaa !8
  %NumOperands97 = getelementptr inbounds %struct.ACItem, ptr %59, i64 0, i32 2
  %60 = load i32, ptr %NumOperands97, align 8, !tbaa !21
  %61 = sext i32 %60 to i64
  %cmp98 = icmp slt i64 %indvars.iv.next252, %61
  br i1 %cmp98, label %for.body101, label %for.cond.cleanup100, !llvm.loop !32

return:                                           ; preds = %for.cond.cleanup, %for.cond.cleanup100, %entry
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local ptr @fACComputeAC(ptr noundef %ResultVar, ptr noundef %OperandNames, ptr noundef %OperandValues, i32 noundef %F, ptr noundef %FileName, i32 noundef %LineNumber) local_unnamed_addr #5 {
entry:
  %Item = alloca %struct.ACItem, align 8
  %0 = add i32 %F, -17
  %switch.i.i = icmp ult i32 %0, -13
  br i1 %switch.i.i, label %if.end.i, label %fACFuncHasXNumOperands.exit

if.end.i:                                         ; preds = %entry
  %1 = icmp ugt i32 %F, 3
  br i1 %1, label %if.end4.i, label %fACFuncHasXNumOperands.exit

if.end4.i:                                        ; preds = %if.end.i
  %cmp.i.not.i = icmp eq i32 %F, 17
  %..i = select i1 %cmp.i.not.i, i32 3, i32 0
  br label %fACFuncHasXNumOperands.exit

fACFuncHasXNumOperands.exit:                      ; preds = %entry, %if.end.i, %if.end4.i
  %retval.0.i = phi i32 [ 1, %entry ], [ 2, %if.end.i ], [ %..i, %if.end4.i ]
  call void @llvm.lifetime.start.p0(i64 72, ptr nonnull %Item) #21
  %2 = load i32, ptr @ACItemCounter, align 4, !tbaa !16
  store i32 %2, ptr %Item, align 8, !tbaa !18
  %F1 = getelementptr inbounds %struct.ACItem, ptr %Item, i64 0, i32 1
  store i32 %F, ptr %F1, align 4, !tbaa !20
  %NumOperands2 = getelementptr inbounds %struct.ACItem, ptr %Item, i64 0, i32 2
  store i32 %retval.0.i, ptr %NumOperands2, align 8, !tbaa !21
  %ResultVar3 = getelementptr inbounds %struct.ACItem, ptr %Item, i64 0, i32 3
  store ptr %ResultVar, ptr %ResultVar3, align 8, !tbaa !22
  %OperandNames4 = getelementptr inbounds %struct.ACItem, ptr %Item, i64 0, i32 4
  store ptr %OperandNames, ptr %OperandNames4, align 8, !tbaa !26
  %OperandValues5 = getelementptr inbounds %struct.ACItem, ptr %Item, i64 0, i32 5
  store ptr %OperandValues, ptr %OperandValues5, align 8, !tbaa !27
  %3 = shl nuw nsw i32 %retval.0.i, 3
  %mul = zext i32 %3 to i64
  %call7 = call noalias ptr @malloc(i64 noundef %mul) #22
  %ACWRTOperands = getelementptr inbounds %struct.ACItem, ptr %Item, i64 0, i32 6
  store ptr %call7, ptr %ACWRTOperands, align 8, !tbaa !23
  %cmp = icmp eq ptr %call7, null
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %fACFuncHasXNumOperands.exit
  %call9 = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.40)
  call void @exit(i32 noundef 1) #23
  unreachable

if.end:                                           ; preds = %fACFuncHasXNumOperands.exit
  %call13 = call noalias ptr @malloc(i64 noundef %mul) #22
  %ACStrings = getelementptr inbounds %struct.ACItem, ptr %Item, i64 0, i32 7
  store ptr %call13, ptr %ACStrings, align 8, !tbaa !28
  %cmp14 = icmp eq ptr %call13, null
  br i1 %cmp14, label %if.then16, label %if.end18

if.then16:                                        ; preds = %if.end
  %call17 = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.40)
  call void @exit(i32 noundef 1) #23
  unreachable

if.end18:                                         ; preds = %if.end
  switch i32 %F, label %sw.default [
    i32 0, label %sw.bb
    i32 1, label %sw.bb36
    i32 2, label %sw.bb56
    i32 3, label %sw.bb67
    i32 4, label %sw.bb78
    i32 5, label %sw.bb91
    i32 6, label %sw.bb101
    i32 7, label %sw.bb111
    i32 8, label %sw.bb126
    i32 9, label %sw.bb141
    i32 10, label %sw.bb154
    i32 11, label %sw.bb167
    i32 12, label %sw.bb177
    i32 13, label %sw.bb190
    i32 14, label %sw.bb197
    i32 15, label %sw.bb206
    i32 16, label %sw.bb212
    i32 17, label %sw.bb218
  ]

sw.bb:                                            ; preds = %if.end18
  %4 = load double, ptr %OperandValues, align 8, !tbaa !10
  %arrayidx20 = getelementptr inbounds double, ptr %OperandValues, i64 1
  %5 = load double, ptr %arrayidx20, align 8, !tbaa !10
  %add = fadd double %4, %5
  %div = fdiv double %4, %add
  %6 = call double @llvm.fabs.f64(double %div)
  store double %6, ptr %call7, align 8, !tbaa !10
  %div27 = fdiv double %5, %add
  %7 = call double @llvm.fabs.f64(double %div27)
  %arrayidx29 = getelementptr inbounds double, ptr %call7, i64 1
  store double %7, ptr %arrayidx29, align 8, !tbaa !10
  %call30 = call ptr @fACDumpAtomicConditionString(ptr noundef %OperandNames, ptr noundef nonnull %OperandValues, i32 noundef 0, i32 noundef 0)
  %8 = load ptr, ptr %ACStrings, align 8, !tbaa !28
  store ptr %call30, ptr %8, align 8, !tbaa !8
  %call33 = call ptr @fACDumpAtomicConditionString(ptr noundef %OperandNames, ptr noundef nonnull %OperandValues, i32 noundef 0, i32 noundef 1)
  %9 = load ptr, ptr %ACStrings, align 8, !tbaa !28
  %arrayidx35 = getelementptr inbounds ptr, ptr %9, i64 1
  store ptr %call33, ptr %arrayidx35, align 8, !tbaa !8
  br label %sw.epilog

sw.bb36:                                          ; preds = %if.end18
  %10 = load double, ptr %OperandValues, align 8, !tbaa !10
  %arrayidx39 = getelementptr inbounds double, ptr %OperandValues, i64 1
  %11 = load double, ptr %arrayidx39, align 8, !tbaa !10
  %sub = fsub double %10, %11
  %div40 = fdiv double %10, %sub
  %12 = call double @llvm.fabs.f64(double %div40)
  store double %12, ptr %call7, align 8, !tbaa !10
  %sub46 = fsub double %11, %10
  %div47 = fdiv double %11, %sub46
  %13 = call double @llvm.fabs.f64(double %div47)
  %arrayidx49 = getelementptr inbounds double, ptr %call7, i64 1
  store double %13, ptr %arrayidx49, align 8, !tbaa !10
  %call50 = call ptr @fACDumpAtomicConditionString(ptr noundef %OperandNames, ptr noundef nonnull %OperandValues, i32 noundef 1, i32 noundef 0)
  %14 = load ptr, ptr %ACStrings, align 8, !tbaa !28
  store ptr %call50, ptr %14, align 8, !tbaa !8
  %call53 = call ptr @fACDumpAtomicConditionString(ptr noundef %OperandNames, ptr noundef nonnull %OperandValues, i32 noundef 1, i32 noundef 1)
  %15 = load ptr, ptr %ACStrings, align 8, !tbaa !28
  %arrayidx55 = getelementptr inbounds ptr, ptr %15, i64 1
  store ptr %call53, ptr %arrayidx55, align 8, !tbaa !8
  br label %sw.epilog

sw.bb56:                                          ; preds = %if.end18
  %arrayidx58 = getelementptr inbounds double, ptr %call7, i64 1
  store double 1.000000e+00, ptr %arrayidx58, align 8, !tbaa !10
  store double 1.000000e+00, ptr %call7, align 8, !tbaa !10
  %call61 = call ptr @fACDumpAtomicConditionString(ptr noundef %OperandNames, ptr noundef %OperandValues, i32 noundef 2, i32 noundef 0)
  %16 = load ptr, ptr %ACStrings, align 8, !tbaa !28
  store ptr %call61, ptr %16, align 8, !tbaa !8
  %call64 = call ptr @fACDumpAtomicConditionString(ptr noundef %OperandNames, ptr noundef %OperandValues, i32 noundef 2, i32 noundef 1)
  %17 = load ptr, ptr %ACStrings, align 8, !tbaa !28
  %arrayidx66 = getelementptr inbounds ptr, ptr %17, i64 1
  store ptr %call64, ptr %arrayidx66, align 8, !tbaa !8
  br label %sw.epilog

sw.bb67:                                          ; preds = %if.end18
  %arrayidx69 = getelementptr inbounds double, ptr %call7, i64 1
  store double 1.000000e+00, ptr %arrayidx69, align 8, !tbaa !10
  store double 1.000000e+00, ptr %call7, align 8, !tbaa !10
  %call72 = call ptr @fACDumpAtomicConditionString(ptr noundef %OperandNames, ptr noundef %OperandValues, i32 noundef 3, i32 noundef 0)
  %18 = load ptr, ptr %ACStrings, align 8, !tbaa !28
  store ptr %call72, ptr %18, align 8, !tbaa !8
  %call75 = call ptr @fACDumpAtomicConditionString(ptr noundef %OperandNames, ptr noundef %OperandValues, i32 noundef 3, i32 noundef 1)
  %19 = load ptr, ptr %ACStrings, align 8, !tbaa !28
  %arrayidx77 = getelementptr inbounds ptr, ptr %19, i64 1
  store ptr %call75, ptr %arrayidx77, align 8, !tbaa !8
  br label %sw.epilog

sw.bb78:                                          ; preds = %if.end18
  %20 = load double, ptr %OperandValues, align 8, !tbaa !10
  %call81 = call double @cos(double noundef %20) #21
  %21 = load double, ptr %OperandValues, align 8, !tbaa !10
  %call83 = call double @sin(double noundef %21) #21
  %div84 = fdiv double %call81, %call83
  %mul85 = fmul double %20, %div84
  %22 = call double @llvm.fabs.f64(double %mul85)
  %23 = load ptr, ptr %ACWRTOperands, align 8, !tbaa !23
  store double %22, ptr %23, align 8, !tbaa !10
  %call88 = call ptr @fACDumpAtomicConditionString(ptr noundef %OperandNames, ptr noundef nonnull %OperandValues, i32 noundef 4, i32 noundef 0)
  %24 = load ptr, ptr %ACStrings, align 8, !tbaa !28
  store ptr %call88, ptr %24, align 8, !tbaa !8
  br label %sw.epilog

sw.bb91:                                          ; preds = %if.end18
  %25 = load double, ptr %OperandValues, align 8, !tbaa !10
  %call94 = call double @tan(double noundef %25) #21
  %mul95 = fmul double %25, %call94
  %26 = call double @llvm.fabs.f64(double %mul95)
  %27 = load ptr, ptr %ACWRTOperands, align 8, !tbaa !23
  store double %26, ptr %27, align 8, !tbaa !10
  %call98 = call ptr @fACDumpAtomicConditionString(ptr noundef %OperandNames, ptr noundef nonnull %OperandValues, i32 noundef 5, i32 noundef 0)
  %28 = load ptr, ptr %ACStrings, align 8, !tbaa !28
  store ptr %call98, ptr %28, align 8, !tbaa !8
  br label %sw.epilog

sw.bb101:                                         ; preds = %if.end18
  %29 = load double, ptr %OperandValues, align 8, !tbaa !10
  %call104 = call double @sin(double noundef %29) #21
  %30 = load double, ptr %OperandValues, align 8, !tbaa !10
  %call106 = call double @cos(double noundef %30) #21
  %mul107 = fmul double %call104, %call106
  %div108 = fdiv double %29, %mul107
  %31 = call double @llvm.fabs.f64(double %div108)
  %32 = load ptr, ptr %ACWRTOperands, align 8, !tbaa !23
  store double %31, ptr %32, align 8, !tbaa !10
  br label %sw.epilog

sw.bb111:                                         ; preds = %if.end18
  %33 = load double, ptr %OperandValues, align 8, !tbaa !10
  %square386 = fmul double %33, %33
  %sub115 = fsub double 1.000000e+00, %square386
  %call116 = call double @sqrt(double noundef %sub115) #21
  %34 = load double, ptr %OperandValues, align 8, !tbaa !10
  %call118 = call double @asin(double noundef %34) #21
  %mul119 = fmul double %call116, %call118
  %div120 = fdiv double %33, %mul119
  %35 = call double @llvm.fabs.f64(double %div120)
  %36 = load ptr, ptr %ACWRTOperands, align 8, !tbaa !23
  store double %35, ptr %36, align 8, !tbaa !10
  %call123 = call ptr @fACDumpAtomicConditionString(ptr noundef %OperandNames, ptr noundef nonnull %OperandValues, i32 noundef 7, i32 noundef 0)
  %37 = load ptr, ptr %ACStrings, align 8, !tbaa !28
  store ptr %call123, ptr %37, align 8, !tbaa !8
  br label %sw.epilog

sw.bb126:                                         ; preds = %if.end18
  %38 = load double, ptr %OperandValues, align 8, !tbaa !10
  %fneg = fneg double %38
  %square385 = fmul double %38, %38
  %sub130 = fsub double 1.000000e+00, %square385
  %call131 = call double @sqrt(double noundef %sub130) #21
  %39 = load double, ptr %OperandValues, align 8, !tbaa !10
  %call133 = call double @acos(double noundef %39) #21
  %mul134 = fmul double %call131, %call133
  %div135 = fdiv double %fneg, %mul134
  %40 = call double @llvm.fabs.f64(double %div135)
  %41 = load ptr, ptr %ACWRTOperands, align 8, !tbaa !23
  store double %40, ptr %41, align 8, !tbaa !10
  %call138 = call ptr @fACDumpAtomicConditionString(ptr noundef %OperandNames, ptr noundef nonnull %OperandValues, i32 noundef 8, i32 noundef 0)
  %42 = load ptr, ptr %ACStrings, align 8, !tbaa !28
  store ptr %call138, ptr %42, align 8, !tbaa !8
  br label %sw.epilog

sw.bb141:                                         ; preds = %if.end18
  %43 = load double, ptr %OperandValues, align 8, !tbaa !10
  %square = fmul double %43, %43
  %call146 = call double @atan(double noundef %43) #21
  %44 = fadd double %call146, %square
  %div148 = fdiv double %43, %44
  %45 = call double @llvm.fabs.f64(double %div148)
  %46 = load ptr, ptr %ACWRTOperands, align 8, !tbaa !23
  store double %45, ptr %46, align 8, !tbaa !10
  %call151 = call ptr @fACDumpAtomicConditionString(ptr noundef %OperandNames, ptr noundef nonnull %OperandValues, i32 noundef 9, i32 noundef 0)
  %47 = load ptr, ptr %ACStrings, align 8, !tbaa !28
  store ptr %call151, ptr %47, align 8, !tbaa !8
  br label %sw.epilog

sw.bb154:                                         ; preds = %if.end18
  %48 = load double, ptr %OperandValues, align 8, !tbaa !10
  %call157 = call double @cosh(double noundef %48) #21
  %49 = load double, ptr %OperandValues, align 8, !tbaa !10
  %call159 = call double @sinh(double noundef %49) #21
  %div160 = fdiv double %call157, %call159
  %mul161 = fmul double %48, %div160
  %50 = call double @llvm.fabs.f64(double %mul161)
  %51 = load ptr, ptr %ACWRTOperands, align 8, !tbaa !23
  store double %50, ptr %51, align 8, !tbaa !10
  %call164 = call ptr @fACDumpAtomicConditionString(ptr noundef %OperandNames, ptr noundef nonnull %OperandValues, i32 noundef 10, i32 noundef 0)
  %52 = load ptr, ptr %ACStrings, align 8, !tbaa !28
  store ptr %call164, ptr %52, align 8, !tbaa !8
  br label %sw.epilog

sw.bb167:                                         ; preds = %if.end18
  %53 = load double, ptr %OperandValues, align 8, !tbaa !10
  %call170 = call double @tanh(double noundef %53) #21
  %mul171 = fmul double %53, %call170
  %54 = call double @llvm.fabs.f64(double %mul171)
  %55 = load ptr, ptr %ACWRTOperands, align 8, !tbaa !23
  store double %54, ptr %55, align 8, !tbaa !10
  %call174 = call ptr @fACDumpAtomicConditionString(ptr noundef %OperandNames, ptr noundef nonnull %OperandValues, i32 noundef 11, i32 noundef 0)
  %56 = load ptr, ptr %ACStrings, align 8, !tbaa !28
  store ptr %call174, ptr %56, align 8, !tbaa !8
  br label %sw.epilog

sw.bb177:                                         ; preds = %if.end18
  %57 = load double, ptr %OperandValues, align 8, !tbaa !10
  %call180 = call double @sinh(double noundef %57) #21
  %58 = load double, ptr %OperandValues, align 8, !tbaa !10
  %call182 = call double @cosh(double noundef %58) #21
  %mul183 = fmul double %call180, %call182
  %div184 = fdiv double %57, %mul183
  %59 = call double @llvm.fabs.f64(double %div184)
  %60 = load ptr, ptr %ACWRTOperands, align 8, !tbaa !23
  store double %59, ptr %60, align 8, !tbaa !10
  %call187 = call ptr @fACDumpAtomicConditionString(ptr noundef %OperandNames, ptr noundef nonnull %OperandValues, i32 noundef 12, i32 noundef 0)
  %61 = load ptr, ptr %ACStrings, align 8, !tbaa !28
  store ptr %call187, ptr %61, align 8, !tbaa !8
  br label %sw.epilog

sw.bb190:                                         ; preds = %if.end18
  %62 = load double, ptr %OperandValues, align 8, !tbaa !10
  %63 = call double @llvm.fabs.f64(double %62)
  store double %63, ptr %call7, align 8, !tbaa !10
  %call194 = call ptr @fACDumpAtomicConditionString(ptr noundef %OperandNames, ptr noundef nonnull %OperandValues, i32 noundef 13, i32 noundef 0)
  %64 = load ptr, ptr %ACStrings, align 8, !tbaa !28
  store ptr %call194, ptr %64, align 8, !tbaa !8
  br label %sw.epilog

sw.bb197:                                         ; preds = %if.end18
  %65 = load double, ptr %OperandValues, align 8, !tbaa !10
  %call199 = call double @log(double noundef %65) #21
  %div200 = fdiv double 1.000000e+00, %call199
  %66 = call double @llvm.fabs.f64(double %div200)
  %67 = load ptr, ptr %ACWRTOperands, align 8, !tbaa !23
  store double %66, ptr %67, align 8, !tbaa !10
  %call203 = call ptr @fACDumpAtomicConditionString(ptr noundef %OperandNames, ptr noundef nonnull %OperandValues, i32 noundef 14, i32 noundef 0)
  %68 = load ptr, ptr %ACStrings, align 8, !tbaa !28
  store ptr %call203, ptr %68, align 8, !tbaa !8
  br label %sw.epilog

sw.bb206:                                         ; preds = %if.end18
  store double 5.000000e-01, ptr %call7, align 8, !tbaa !10
  %call209 = call ptr @fACDumpAtomicConditionString(ptr noundef %OperandNames, ptr noundef %OperandValues, i32 noundef 15, i32 noundef 0)
  %69 = load ptr, ptr %ACStrings, align 8, !tbaa !28
  store ptr %call209, ptr %69, align 8, !tbaa !8
  br label %sw.epilog

sw.bb212:                                         ; preds = %if.end18
  store double 1.000000e+00, ptr %call7, align 8, !tbaa !10
  %call215 = call ptr @fACDumpAtomicConditionString(ptr noundef %OperandNames, ptr noundef %OperandValues, i32 noundef 16, i32 noundef 0)
  %70 = load ptr, ptr %ACStrings, align 8, !tbaa !28
  store ptr %call215, ptr %70, align 8, !tbaa !8
  br label %sw.epilog

sw.bb218:                                         ; preds = %if.end18
  %71 = load double, ptr %OperandValues, align 8, !tbaa !10
  %arrayidx220 = getelementptr inbounds double, ptr %OperandValues, i64 1
  %72 = load double, ptr %arrayidx220, align 8, !tbaa !10
  %mul221 = fmul double %71, %72
  %arrayidx225 = getelementptr inbounds double, ptr %OperandValues, i64 2
  %73 = load double, ptr %arrayidx225, align 8, !tbaa !10
  %74 = call double @llvm.fmuladd.f64(double %71, double %72, double %73)
  %div226 = fdiv double %mul221, %74
  %75 = call double @llvm.fabs.f64(double %div226)
  store double %75, ptr %call7, align 8, !tbaa !10
  %arrayidx238 = getelementptr inbounds double, ptr %call7, i64 1
  store double %75, ptr %arrayidx238, align 8, !tbaa !10
  %div244 = fdiv double %73, %74
  %76 = call double @llvm.fabs.f64(double %div244)
  %arrayidx246 = getelementptr inbounds double, ptr %call7, i64 2
  store double %76, ptr %arrayidx246, align 8, !tbaa !10
  %call247 = call ptr @fACDumpAtomicConditionString(ptr noundef %OperandNames, ptr noundef nonnull %OperandValues, i32 noundef 17, i32 noundef 0)
  %77 = load ptr, ptr %ACStrings, align 8, !tbaa !28
  store ptr %call247, ptr %77, align 8, !tbaa !8
  %call250 = call ptr @fACDumpAtomicConditionString(ptr noundef %OperandNames, ptr noundef nonnull %OperandValues, i32 noundef 17, i32 noundef 1)
  %78 = load ptr, ptr %ACStrings, align 8, !tbaa !28
  %arrayidx252 = getelementptr inbounds ptr, ptr %78, i64 1
  store ptr %call250, ptr %arrayidx252, align 8, !tbaa !8
  %call253 = call ptr @fACDumpAtomicConditionString(ptr noundef %OperandNames, ptr noundef nonnull %OperandValues, i32 noundef 17, i32 noundef 2)
  %79 = load ptr, ptr %ACStrings, align 8, !tbaa !28
  %arrayidx255 = getelementptr inbounds ptr, ptr %79, i64 2
  store ptr %call253, ptr %arrayidx255, align 8, !tbaa !8
  br label %sw.epilog

sw.default:                                       ; preds = %if.end18
  %puts = call i32 @puts(ptr nonnull @str)
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.default, %sw.bb218, %sw.bb212, %sw.bb206, %sw.bb197, %sw.bb190, %sw.bb177, %sw.bb167, %sw.bb154, %sw.bb141, %sw.bb126, %sw.bb111, %sw.bb101, %sw.bb91, %sw.bb78, %sw.bb67, %sw.bb56, %sw.bb36, %sw.bb
  %FileName257 = getelementptr inbounds %struct.ACItem, ptr %Item, i64 0, i32 8
  store ptr %FileName, ptr %FileName257, align 8, !tbaa !24
  %LineNumber258 = getelementptr inbounds %struct.ACItem, ptr %Item, i64 0, i32 9
  store i32 %LineNumber, ptr %LineNumber258, align 8, !tbaa !25
  %80 = load ptr, ptr @ACs, align 8, !tbaa !8
  call void @fACSetACItem(ptr noundef %80, ptr noundef nonnull %Item)
  %81 = load ptr, ptr @ACs, align 8, !tbaa !8
  %ACItems = getelementptr inbounds %struct.ACTable, ptr %81, i64 0, i32 1
  %82 = load ptr, ptr %ACItems, align 8, !tbaa !12
  %83 = load i32, ptr %Item, align 8, !tbaa !18
  %idxprom = sext i32 %83 to i64
  %arrayidx260 = getelementptr inbounds ptr, ptr %82, i64 %idxprom
  call void @llvm.lifetime.end.p0(i64 72, ptr nonnull %Item) #21
  ret ptr %arrayidx260
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn
declare double @llvm.fabs.f64(double) #11

; Function Attrs: mustprogress nofree nounwind willreturn writeonly
declare double @cos(double noundef) local_unnamed_addr #12

; Function Attrs: mustprogress nofree nounwind willreturn writeonly
declare double @sin(double noundef) local_unnamed_addr #12

; Function Attrs: mustprogress nofree nounwind willreturn writeonly
declare double @tan(double noundef) local_unnamed_addr #12

; Function Attrs: mustprogress nofree nounwind willreturn writeonly
declare double @sqrt(double noundef) local_unnamed_addr #12

; Function Attrs: mustprogress nofree nounwind willreturn writeonly
declare double @asin(double noundef) local_unnamed_addr #12

; Function Attrs: mustprogress nofree nounwind willreturn writeonly
declare double @acos(double noundef) local_unnamed_addr #12

; Function Attrs: mustprogress nofree nounwind willreturn writeonly
declare double @atan(double noundef) local_unnamed_addr #12

; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn
declare double @llvm.fmuladd.f64(double, double, double) #11

; Function Attrs: mustprogress nofree nounwind willreturn writeonly
declare double @cosh(double noundef) local_unnamed_addr #12

; Function Attrs: mustprogress nofree nounwind willreturn writeonly
declare double @sinh(double noundef) local_unnamed_addr #12

; Function Attrs: mustprogress nofree nounwind willreturn writeonly
declare double @tanh(double noundef) local_unnamed_addr #12

; Function Attrs: mustprogress nofree nounwind willreturn writeonly
declare double @log(double noundef) local_unnamed_addr #12

; Function Attrs: nounwind uwtable
define dso_local void @fACStoreACs() local_unnamed_addr #5 {
entry:
  %File = alloca [5000 x i8], align 16
  %puts = call i32 @puts(ptr nonnull @str.82)
  call void @llvm.lifetime.start.p0(i64 5000, ptr nonnull %File) #21
  call void @fAFGenerateFileString(ptr noundef nonnull %File, ptr noundef nonnull @.str.43, ptr noundef nonnull @.str.44)
  %call2 = call noalias ptr @fopen(ptr noundef nonnull %File, ptr noundef nonnull @.str.45)
  %0 = call i64 @fwrite(ptr nonnull @.str.46, i64 2, i64 1, ptr %call2)
  %1 = call i64 @fwrite(ptr nonnull @.str.47, i64 10, i64 1, ptr %call2)
  br label %while.body

while.body:                                       ; preds = %entry, %if.end80
  %indvars.iv150 = phi i64 [ 0, %entry ], [ %indvars.iv.next151, %if.end80 ]
  %RecordsStored.0140 = phi i64 [ 0, %entry ], [ %RecordsStored.1, %if.end80 ]
  %2 = load ptr, ptr @ACs, align 8, !tbaa !8
  %ACItems = getelementptr inbounds %struct.ACTable, ptr %2, i64 0, i32 1
  %3 = load ptr, ptr %ACItems, align 8, !tbaa !12
  %arrayidx = getelementptr inbounds ptr, ptr %3, i64 %indvars.iv150
  %4 = load ptr, ptr %arrayidx, align 8, !tbaa !8
  %cmp5.not = icmp eq ptr %4, null
  br i1 %cmp5.not, label %if.end80, label %if.then

if.then:                                          ; preds = %while.body
  %5 = load i32, ptr %4, align 8, !tbaa !18
  %F = getelementptr inbounds %struct.ACItem, ptr %4, i64 0, i32 1
  %6 = load i32, ptr %F, align 4, !tbaa !20
  %ResultVar = getelementptr inbounds %struct.ACItem, ptr %4, i64 0, i32 3
  %7 = load ptr, ptr %ResultVar, align 8, !tbaa !22
  %call15 = call i32 (ptr, ptr, ...) @fprintf(ptr noundef %call2, ptr noundef nonnull @.str.48, i32 noundef %5, i32 noundef %6, ptr noundef %7)
  %8 = load ptr, ptr @ACs, align 8, !tbaa !8
  %ACItems16125 = getelementptr inbounds %struct.ACTable, ptr %8, i64 0, i32 1
  %9 = load ptr, ptr %ACItems16125, align 8, !tbaa !12
  %arrayidx18126 = getelementptr inbounds ptr, ptr %9, i64 %indvars.iv150
  %10 = load ptr, ptr %arrayidx18126, align 8, !tbaa !8
  %NumOperands127 = getelementptr inbounds %struct.ACItem, ptr %10, i64 0, i32 2
  %11 = load i32, ptr %NumOperands127, align 8, !tbaa !21
  %cmp19128 = icmp sgt i32 %11, 0
  br i1 %cmp19128, label %for.body, label %for.cond32.preheader

for.cond32.preheader:                             ; preds = %for.body, %if.then
  %12 = load ptr, ptr @ACs, align 8, !tbaa !8
  %ACItems33130 = getelementptr inbounds %struct.ACTable, ptr %12, i64 0, i32 1
  %13 = load ptr, ptr %ACItems33130, align 8, !tbaa !12
  %arrayidx35131 = getelementptr inbounds ptr, ptr %13, i64 %indvars.iv150
  %14 = load ptr, ptr %arrayidx35131, align 8, !tbaa !8
  %NumOperands36132 = getelementptr inbounds %struct.ACItem, ptr %14, i64 0, i32 2
  %15 = load i32, ptr %NumOperands36132, align 8, !tbaa !21
  %cmp37133 = icmp sgt i32 %15, 0
  br i1 %cmp37133, label %for.body39, label %for.cond50.preheader

for.body:                                         ; preds = %if.then, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %if.then ]
  %16 = phi ptr [ %24, %for.body ], [ %10, %if.then ]
  %OperandNames = getelementptr inbounds %struct.ACItem, ptr %16, i64 0, i32 4
  %17 = load ptr, ptr %OperandNames, align 8, !tbaa !26
  %arrayidx24 = getelementptr inbounds ptr, ptr %17, i64 %indvars.iv
  %18 = load ptr, ptr %arrayidx24, align 8, !tbaa !8
  %OperandValues = getelementptr inbounds %struct.ACItem, ptr %16, i64 0, i32 5
  %19 = load ptr, ptr %OperandValues, align 8, !tbaa !27
  %arrayidx29 = getelementptr inbounds double, ptr %19, i64 %indvars.iv
  %20 = load double, ptr %arrayidx29, align 8, !tbaa !10
  %21 = trunc i64 %indvars.iv to i32
  %call30 = call i32 (ptr, ptr, ...) @fprintf(ptr noundef %call2, ptr noundef nonnull @.str.49, i32 noundef %21, ptr noundef %18, i32 noundef %21, double noundef %20)
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %22 = load ptr, ptr @ACs, align 8, !tbaa !8
  %ACItems16 = getelementptr inbounds %struct.ACTable, ptr %22, i64 0, i32 1
  %23 = load ptr, ptr %ACItems16, align 8, !tbaa !12
  %arrayidx18 = getelementptr inbounds ptr, ptr %23, i64 %indvars.iv150
  %24 = load ptr, ptr %arrayidx18, align 8, !tbaa !8
  %NumOperands = getelementptr inbounds %struct.ACItem, ptr %24, i64 0, i32 2
  %25 = load i32, ptr %NumOperands, align 8, !tbaa !21
  %26 = sext i32 %25 to i64
  %cmp19 = icmp slt i64 %indvars.iv.next, %26
  br i1 %cmp19, label %for.body, label %for.cond32.preheader, !llvm.loop !33

for.cond50.preheader:                             ; preds = %for.body39, %for.cond32.preheader
  %27 = load ptr, ptr @ACs, align 8, !tbaa !8
  %ACItems51135 = getelementptr inbounds %struct.ACTable, ptr %27, i64 0, i32 1
  %28 = load ptr, ptr %ACItems51135, align 8, !tbaa !12
  %arrayidx53136 = getelementptr inbounds ptr, ptr %28, i64 %indvars.iv150
  %29 = load ptr, ptr %arrayidx53136, align 8, !tbaa !8
  %NumOperands54137 = getelementptr inbounds %struct.ACItem, ptr %29, i64 0, i32 2
  %30 = load i32, ptr %NumOperands54137, align 8, !tbaa !21
  %cmp55138 = icmp sgt i32 %30, 0
  br i1 %cmp55138, label %for.body57, label %for.cond.cleanup56

for.body39:                                       ; preds = %for.cond32.preheader, %for.body39
  %indvars.iv144 = phi i64 [ %indvars.iv.next145, %for.body39 ], [ 0, %for.cond32.preheader ]
  %31 = phi ptr [ %37, %for.body39 ], [ %14, %for.cond32.preheader ]
  %ACWRTOperands = getelementptr inbounds %struct.ACItem, ptr %31, i64 0, i32 6
  %32 = load ptr, ptr %ACWRTOperands, align 8, !tbaa !23
  %arrayidx44 = getelementptr inbounds double, ptr %32, i64 %indvars.iv144
  %33 = load double, ptr %arrayidx44, align 8, !tbaa !10
  %34 = trunc i64 %indvars.iv144 to i32
  %call45 = call i32 (ptr, ptr, ...) @fprintf(ptr noundef %call2, ptr noundef nonnull @.str.50, i32 noundef %34, double noundef %33)
  %indvars.iv.next145 = add nuw nsw i64 %indvars.iv144, 1
  %35 = load ptr, ptr @ACs, align 8, !tbaa !8
  %ACItems33 = getelementptr inbounds %struct.ACTable, ptr %35, i64 0, i32 1
  %36 = load ptr, ptr %ACItems33, align 8, !tbaa !12
  %arrayidx35 = getelementptr inbounds ptr, ptr %36, i64 %indvars.iv150
  %37 = load ptr, ptr %arrayidx35, align 8, !tbaa !8
  %NumOperands36 = getelementptr inbounds %struct.ACItem, ptr %37, i64 0, i32 2
  %38 = load i32, ptr %NumOperands36, align 8, !tbaa !21
  %39 = sext i32 %38 to i64
  %cmp37 = icmp slt i64 %indvars.iv.next145, %39
  br i1 %cmp37, label %for.body39, label %for.cond50.preheader, !llvm.loop !34

for.cond.cleanup56:                               ; preds = %for.body57, %for.cond50.preheader
  %.lcssa = phi ptr [ %29, %for.cond50.preheader ], [ %53, %for.body57 ]
  %FileName = getelementptr inbounds %struct.ACItem, ptr %.lcssa, i64 0, i32 8
  %40 = load ptr, ptr %FileName, align 8, !tbaa !24
  %call70 = call i32 (ptr, ptr, ...) @fprintf(ptr noundef %call2, ptr noundef nonnull @.str.52, ptr noundef %40)
  %41 = load ptr, ptr @ACs, align 8, !tbaa !8
  %ACItems71 = getelementptr inbounds %struct.ACTable, ptr %41, i64 0, i32 1
  %42 = load ptr, ptr %ACItems71, align 8, !tbaa !12
  %arrayidx73 = getelementptr inbounds ptr, ptr %42, i64 %indvars.iv150
  %43 = load ptr, ptr %arrayidx73, align 8, !tbaa !8
  %LineNumber = getelementptr inbounds %struct.ACItem, ptr %43, i64 0, i32 9
  %44 = load i32, ptr %LineNumber, align 8, !tbaa !25
  %call74 = call i32 (ptr, ptr, ...) @fprintf(ptr noundef %call2, ptr noundef nonnull @.str.53, i32 noundef %44)
  %inc75 = add i64 %RecordsStored.0140, 1
  %45 = load ptr, ptr @ACs, align 8, !tbaa !8
  %46 = load i64, ptr %45, align 8, !tbaa !15
  %cmp76.not = icmp eq i64 %inc75, %46
  br i1 %cmp76.not, label %if.else, label %if.then77

for.body57:                                       ; preds = %for.cond50.preheader, %for.body57
  %indvars.iv147 = phi i64 [ %indvars.iv.next148, %for.body57 ], [ 0, %for.cond50.preheader ]
  %47 = phi ptr [ %53, %for.body57 ], [ %29, %for.cond50.preheader ]
  %ACStrings = getelementptr inbounds %struct.ACItem, ptr %47, i64 0, i32 7
  %48 = load ptr, ptr %ACStrings, align 8, !tbaa !28
  %arrayidx62 = getelementptr inbounds ptr, ptr %48, i64 %indvars.iv147
  %49 = load ptr, ptr %arrayidx62, align 8, !tbaa !8
  %50 = trunc i64 %indvars.iv147 to i32
  %call63 = call i32 (ptr, ptr, ...) @fprintf(ptr noundef %call2, ptr noundef nonnull @.str.51, i32 noundef %50, ptr noundef %49)
  %indvars.iv.next148 = add nuw nsw i64 %indvars.iv147, 1
  %51 = load ptr, ptr @ACs, align 8, !tbaa !8
  %ACItems51 = getelementptr inbounds %struct.ACTable, ptr %51, i64 0, i32 1
  %52 = load ptr, ptr %ACItems51, align 8, !tbaa !12
  %arrayidx53 = getelementptr inbounds ptr, ptr %52, i64 %indvars.iv150
  %53 = load ptr, ptr %arrayidx53, align 8, !tbaa !8
  %NumOperands54 = getelementptr inbounds %struct.ACItem, ptr %53, i64 0, i32 2
  %54 = load i32, ptr %NumOperands54, align 8, !tbaa !21
  %55 = sext i32 %54 to i64
  %cmp55 = icmp slt i64 %indvars.iv.next148, %55
  br i1 %cmp55, label %for.body57, label %for.cond.cleanup56, !llvm.loop !35

if.then77:                                        ; preds = %for.cond.cleanup56
  %56 = call i64 @fwrite(ptr nonnull @.str.54, i64 5, i64 1, ptr %call2)
  br label %if.end80

if.else:                                          ; preds = %for.cond.cleanup56
  %57 = call i64 @fwrite(ptr nonnull @.str.55, i64 4, i64 1, ptr %call2)
  br label %if.end80

if.end80:                                         ; preds = %if.then77, %if.else, %while.body
  %RecordsStored.1 = phi i64 [ %inc75, %if.then77 ], [ %inc75, %if.else ], [ %RecordsStored.0140, %while.body ]
  %indvars.iv.next151 = add nuw nsw i64 %indvars.iv150, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next151, 1000000
  br i1 %exitcond.not, label %while.end, label %while.body, !llvm.loop !36

while.end:                                        ; preds = %if.end80
  %58 = call i64 @fwrite(ptr nonnull @.str.56, i64 3, i64 1, ptr %call2)
  %59 = call i64 @fwrite(ptr nonnull @.str.57, i64 2, i64 1, ptr %call2)
  %call84 = call i32 @fclose(ptr noundef %call2)
  %call86 = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.58, ptr noundef nonnull %File)
  call void @llvm.lifetime.end.p0(i64 5000, ptr nonnull %File) #21
  ret void
}

; Function Attrs: nofree nounwind
declare noalias noundef ptr @fopen(ptr nocapture noundef readonly, ptr nocapture noundef readonly) local_unnamed_addr #3

; Function Attrs: nofree nounwind
declare noundef i32 @fprintf(ptr nocapture noundef, ptr nocapture noundef readonly, ...) local_unnamed_addr #3

; Function Attrs: nofree nounwind
declare noundef i32 @fclose(ptr nocapture noundef) local_unnamed_addr #3

; Function Attrs: nounwind uwtable
define dso_local void @fAFCreateAFComponent(ptr nocapture noundef writeonly %AddressToAllocateAt) local_unnamed_addr #5 {
entry:
  %call = call noalias dereferenceable_or_null(48) ptr @malloc(i64 noundef 48) #22
  store ptr %call, ptr %AddressToAllocateAt, align 8, !tbaa !8
  %cmp = icmp eq ptr %call, null
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call1 = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.59)
  call void @exit(i32 noundef 1) #23
  unreachable

if.end:                                           ; preds = %entry
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local void @fAFCreateAFItem(ptr nocapture noundef writeonly %AddressToAllocateAt) local_unnamed_addr #5 {
entry:
  %call = call noalias dereferenceable_or_null(24) ptr @malloc(i64 noundef 24) #22
  store ptr %call, ptr %AddressToAllocateAt, align 8, !tbaa !8
  %cmp = icmp eq ptr %call, null
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call1 = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.60)
  call void @exit(i32 noundef 1) #23
  unreachable

if.end:                                           ; preds = %entry
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local void @fAFCreateAFTable(ptr nocapture noundef writeonly %AddressToAllocateAt) local_unnamed_addr #5 {
entry:
  %call = call noalias dereferenceable_or_null(16) ptr @malloc(i64 noundef 16) #22
  store ptr %call, ptr %AddressToAllocateAt, align 8, !tbaa !8
  %cmp = icmp eq ptr %call, null
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call1 = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.60)
  call void @exit(i32 noundef 1) #23
  unreachable

if.end:                                           ; preds = %entry
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @fAFfp32markForResult(float noundef %res) local_unnamed_addr #13 {
entry:
  %res.addr = alloca float, align 4
  store float %res, ptr %res.addr, align 4, !tbaa !37
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @fAFfp64markForResult(double noundef %res) local_unnamed_addr #13 {
entry:
  %res.addr = alloca double, align 8
  store double %res, ptr %res.addr, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn uwtable
define dso_local i32 @min(i32 noundef %a, i32 noundef %b) local_unnamed_addr #14 {
entry:
  %0 = call i32 @llvm.smin.i32(i32 %a, i32 %b)
  ret i32 %0
}

; Function Attrs: nounwind uwtable
define dso_local noalias ptr @fAFFlattenAFComponentsPath(ptr noundef %ProductObject) local_unnamed_addr #5 {
entry:
  %Height = getelementptr inbounds %struct.AFProduct, ptr %ProductObject, i64 0, i32 4
  %0 = load i32, ptr %Height, align 8, !tbaa !39
  %conv = sext i32 %0 to i64
  %mul = shl nsw i64 %conv, 3
  %call = call noalias ptr @malloc(i64 noundef %mul) #22
  %cmp = icmp eq ptr %call, null
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call2 = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.61)
  call void @exit(i32 noundef 1) #23
  unreachable

if.end:                                           ; preds = %entry
  store ptr %ProductObject, ptr %call, align 8, !tbaa !8
  %cmp416 = icmp sgt i32 %0, 1
  br i1 %cmp416, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %if.end
  %wide.trip.count = zext i32 %0 to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %if.end
  ret ptr %call

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 1, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %ProductObjectWalker.017 = phi ptr [ %ProductObject, %for.body.preheader ], [ %1, %for.body ]
  %ProductTail = getelementptr inbounds %struct.AFProduct, ptr %ProductObjectWalker.017, i64 0, i32 2
  %1 = load ptr, ptr %ProductTail, align 8, !tbaa !41
  %arrayidx6 = getelementptr inbounds ptr, ptr %call, i64 %indvars.iv
  store ptr %1, ptr %arrayidx6, align 8, !tbaa !8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !42
}

; Function Attrs: nounwind uwtable
define dso_local ptr @fAFFlattenAllComponentPaths() local_unnamed_addr #5 {
entry:
  %0 = load i32, ptr @AFComponentCounter, align 4, !tbaa !16
  %conv = sext i32 %0 to i64
  %mul = shl nsw i64 %conv, 3
  %call = call noalias ptr @malloc(i64 noundef %mul) #22
  store ptr %call, ptr @Paths, align 8, !tbaa !8
  %cmp = icmp eq ptr %call, null
  br i1 %cmp, label %if.then, label %for.cond.preheader

for.cond.preheader:                               ; preds = %entry
  %1 = load ptr, ptr @AFs, align 8, !tbaa !8
  %2 = load i64, ptr %1, align 8, !tbaa !43
  %cmp361.not = icmp eq i64 %2, 0
  br i1 %cmp361.not, label %for.cond.cleanup, label %for.cond5.preheader

if.then:                                          ; preds = %entry
  %call2 = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.61)
  call void @exit(i32 noundef 1) #23
  unreachable

for.cond5.preheader:                              ; preds = %for.cond.preheader, %for.cond.cleanup8
  %I.062 = phi i64 [ %inc34, %for.cond.cleanup8 ], [ 0, %for.cond.preheader ]
  %3 = load ptr, ptr @AFs, align 8, !tbaa !8
  %AFItems56 = getelementptr inbounds %struct.AFTable, ptr %3, i64 0, i32 1
  %4 = load ptr, ptr %AFItems56, align 8, !tbaa !45
  %arrayidx57 = getelementptr inbounds ptr, ptr %4, i64 %I.062
  %5 = load ptr, ptr %arrayidx57, align 8, !tbaa !8
  %NumAFComponents58 = getelementptr inbounds %struct.AFItem, ptr %5, i64 0, i32 2
  %6 = load i32, ptr %NumAFComponents58, align 8, !tbaa !46
  %cmp659 = icmp sgt i32 %6, 0
  br i1 %cmp659, label %for.body9, label %for.cond.cleanup8

for.cond.cleanup:                                 ; preds = %for.cond.cleanup8, %for.cond.preheader
  %7 = load ptr, ptr @Paths, align 8, !tbaa !8
  ret ptr %7

for.cond.cleanup8:                                ; preds = %for.cond.cleanup23, %for.cond5.preheader
  %inc34 = add nuw i64 %I.062, 1
  %8 = load ptr, ptr @AFs, align 8, !tbaa !8
  %9 = load i64, ptr %8, align 8, !tbaa !43
  %cmp3 = icmp ult i64 %inc34, %9
  br i1 %cmp3, label %for.cond5.preheader, label %for.cond.cleanup, !llvm.loop !48

for.body9:                                        ; preds = %for.cond5.preheader, %for.cond.cleanup23
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.cond.cleanup23 ], [ 0, %for.cond5.preheader ]
  %10 = phi ptr [ %23, %for.cond.cleanup23 ], [ %5, %for.cond5.preheader ]
  %Components = getelementptr inbounds %struct.AFItem, ptr %10, i64 0, i32 1
  %11 = load ptr, ptr %Components, align 8, !tbaa !49
  %arrayidx12 = getelementptr inbounds ptr, ptr %11, i64 %indvars.iv
  %12 = load ptr, ptr %arrayidx12, align 8, !tbaa !8
  %13 = load ptr, ptr @Paths, align 8, !tbaa !8
  %14 = load i32, ptr %12, align 8, !tbaa !50
  %idxprom13 = sext i32 %14 to i64
  %arrayidx14 = getelementptr inbounds ptr, ptr %13, i64 %idxprom13
  store ptr %12, ptr %arrayidx14, align 8, !tbaa !8
  %15 = load ptr, ptr @AFs, align 8, !tbaa !8
  %AFItems1648 = getelementptr inbounds %struct.AFTable, ptr %15, i64 0, i32 1
  %16 = load ptr, ptr %AFItems1648, align 8, !tbaa !45
  %arrayidx1749 = getelementptr inbounds ptr, ptr %16, i64 %I.062
  %17 = load ptr, ptr %arrayidx1749, align 8, !tbaa !8
  %Components1850 = getelementptr inbounds %struct.AFItem, ptr %17, i64 0, i32 1
  %18 = load ptr, ptr %Components1850, align 8, !tbaa !49
  %arrayidx2051 = getelementptr inbounds ptr, ptr %18, i64 %indvars.iv
  %19 = load ptr, ptr %arrayidx2051, align 8, !tbaa !8
  %Height52 = getelementptr inbounds %struct.AFProduct, ptr %19, i64 0, i32 4
  %20 = load i32, ptr %Height52, align 8, !tbaa !39
  %cmp2153 = icmp sgt i32 %20, 1
  br i1 %cmp2153, label %for.body24, label %for.cond.cleanup23

for.cond.cleanup23:                               ; preds = %for.body24, %for.body9
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %21 = load ptr, ptr @AFs, align 8, !tbaa !8
  %AFItems = getelementptr inbounds %struct.AFTable, ptr %21, i64 0, i32 1
  %22 = load ptr, ptr %AFItems, align 8, !tbaa !45
  %arrayidx = getelementptr inbounds ptr, ptr %22, i64 %I.062
  %23 = load ptr, ptr %arrayidx, align 8, !tbaa !8
  %NumAFComponents = getelementptr inbounds %struct.AFItem, ptr %23, i64 0, i32 2
  %24 = load i32, ptr %NumAFComponents, align 8, !tbaa !46
  %25 = sext i32 %24 to i64
  %cmp6 = icmp slt i64 %indvars.iv.next, %25
  br i1 %cmp6, label %for.body9, label %for.cond.cleanup8, !llvm.loop !51

for.body24:                                       ; preds = %for.body9, %for.body24
  %K.055 = phi i32 [ %inc, %for.body24 ], [ 1, %for.body9 ]
  %ProductObjectWalker.054 = phi ptr [ %29, %for.body24 ], [ %12, %for.body9 ]
  %ProductTail = getelementptr inbounds %struct.AFProduct, ptr %ProductObjectWalker.054, i64 0, i32 2
  %26 = load ptr, ptr %ProductTail, align 8, !tbaa !41
  %27 = load ptr, ptr @Paths, align 8, !tbaa !8
  %28 = load i32, ptr %26, align 8, !tbaa !50
  %idxprom27 = sext i32 %28 to i64
  %arrayidx28 = getelementptr inbounds ptr, ptr %27, i64 %idxprom27
  store ptr %26, ptr %arrayidx28, align 8, !tbaa !8
  %29 = load ptr, ptr %ProductTail, align 8, !tbaa !41
  %inc = add nuw nsw i32 %K.055, 1
  %30 = load ptr, ptr @AFs, align 8, !tbaa !8
  %AFItems16 = getelementptr inbounds %struct.AFTable, ptr %30, i64 0, i32 1
  %31 = load ptr, ptr %AFItems16, align 8, !tbaa !45
  %arrayidx17 = getelementptr inbounds ptr, ptr %31, i64 %I.062
  %32 = load ptr, ptr %arrayidx17, align 8, !tbaa !8
  %Components18 = getelementptr inbounds %struct.AFItem, ptr %32, i64 0, i32 1
  %33 = load ptr, ptr %Components18, align 8, !tbaa !49
  %arrayidx20 = getelementptr inbounds ptr, ptr %33, i64 %indvars.iv
  %34 = load ptr, ptr %arrayidx20, align 8, !tbaa !8
  %Height = getelementptr inbounds %struct.AFProduct, ptr %34, i64 0, i32 4
  %35 = load i32, ptr %Height, align 8, !tbaa !39
  %cmp21 = icmp slt i32 %inc, %35
  br i1 %cmp21, label %for.body24, label %for.cond.cleanup23, !llvm.loop !52
}

; Function Attrs: argmemonly mustprogress nofree nounwind readonly willreturn uwtable
define dso_local i32 @fAFisMemoryOpInstruction(ptr noundef readonly %InstructionString) local_unnamed_addr #15 {
entry:
  %call = call ptr @strstr(ptr noundef nonnull dereferenceable(1) %InstructionString, ptr noundef nonnull @.str.62) #24
  %cmp.not = icmp eq ptr %call, null
  br i1 %cmp.not, label %lor.lhs.false, label %return

lor.lhs.false:                                    ; preds = %entry
  %call1 = call ptr @strstr(ptr noundef nonnull dereferenceable(1) %InstructionString, ptr noundef nonnull @.str.63) #24
  %cmp2.not = icmp ne ptr %call1, null
  %spec.select = zext i1 %cmp2.not to i32
  br label %return

return:                                           ; preds = %lor.lhs.false, %entry
  %retval.0 = phi i32 [ 1, %entry ], [ %spec.select, %lor.lhs.false ]
  ret i32 %retval.0
}

; Function Attrs: argmemonly mustprogress nofree nounwind readonly willreturn
declare ptr @strstr(ptr noundef, ptr nocapture noundef) local_unnamed_addr #8

; Function Attrs: mustprogress nofree norecurse nosync nounwind readonly willreturn uwtable
define dso_local i32 @fAFComparator(ptr nocapture noundef readonly %a, ptr nocapture noundef readonly %b) #16 {
entry:
  %0 = load ptr, ptr %a, align 8, !tbaa !8
  %AF = getelementptr inbounds %struct.AFProduct, ptr %0, i64 0, i32 5
  %1 = load double, ptr %AF, align 8, !tbaa !53
  %2 = load ptr, ptr %b, align 8, !tbaa !8
  %AF3 = getelementptr inbounds %struct.AFProduct, ptr %2, i64 0, i32 5
  %3 = load double, ptr %AF3, align 8, !tbaa !53
  %cmp = fcmp ogt double %3, %1
  %cmp4 = fcmp une double %3, %1
  %. = sext i1 %cmp4 to i32
  %retval.0 = select i1 %cmp, i32 1, i32 %.
  ret i32 %retval.0
}

; Function Attrs: nounwind uwtable
define dso_local void @fAFInitialize() local_unnamed_addr #5 {
entry:
  store i32 0, ptr @AFItemCounter, align 4, !tbaa !16
  store i32 0, ptr @AFComponentCounter, align 4, !tbaa !16
  %call.i = call noalias dereferenceable_or_null(16) ptr @malloc(i64 noundef 16) #22
  store ptr %call.i, ptr @AFs, align 8, !tbaa !8
  %cmp.i = icmp eq ptr %call.i, null
  br i1 %cmp.i, label %if.then.i, label %fAFCreateAFTable.exit

if.then.i:                                        ; preds = %entry
  %call1.i = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.60) #21
  call void @exit(i32 noundef 1) #23
  unreachable

fAFCreateAFTable.exit:                            ; preds = %entry
  %call = call noalias dereferenceable_or_null(80000) ptr @malloc(i64 noundef 80000) #22
  %AFItems = getelementptr inbounds %struct.AFTable, ptr %call.i, i64 0, i32 1
  store ptr %call, ptr %AFItems, align 8, !tbaa !45
  %cmp = icmp eq ptr %call, null
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %fAFCreateAFTable.exit
  %call1 = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.64)
  call void @exit(i32 noundef 1) #23
  unreachable

if.end:                                           ; preds = %fAFCreateAFTable.exit
  store i64 0, ptr %call.i, align 8, !tbaa !43
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local ptr @fAFComputeAF(ptr nocapture noundef readonly %AC, ptr nocapture noundef readonly %AFItemWRTOperands, i32 noundef %NumOperands) local_unnamed_addr #5 {
entry:
  %cmp151 = icmp sgt i32 %NumOperands, 0
  br i1 %cmp151, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %NumOperands to i64
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.inc
  %phi.cast = sext i32 %TotalAFComponents.1 to i64
  %phi.bo = shl nsw i64 %phi.cast, 3
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  %TotalAFComponents.0.lcssa = phi i64 [ 0, %entry ], [ %phi.bo, %for.cond.cleanup.loopexit ]
  %call.i = call noalias dereferenceable_or_null(24) ptr @malloc(i64 noundef 24) #22
  %cmp.i = icmp eq ptr %call.i, null
  br i1 %cmp.i, label %if.then.i, label %fAFCreateAFItem.exit

if.then.i:                                        ; preds = %for.cond.cleanup
  %call1.i = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.60) #21
  call void @exit(i32 noundef 1) #23
  unreachable

fAFCreateAFItem.exit:                             ; preds = %for.cond.cleanup
  %0 = load i32, ptr @AFItemCounter, align 4, !tbaa !16
  store i32 %0, ptr %call.i, align 8, !tbaa !54
  %NumAFComponents5 = getelementptr inbounds %struct.AFItem, ptr %call.i, i64 0, i32 2
  store i32 0, ptr %NumAFComponents5, align 8, !tbaa !46
  %call = call noalias ptr @malloc(i64 noundef %TotalAFComponents.0.lcssa) #22
  %Components = getelementptr inbounds %struct.AFItem, ptr %call.i, i64 0, i32 1
  store ptr %call, ptr %Components, align 8, !tbaa !49
  %cmp6 = icmp eq ptr %call, null
  br i1 %cmp6, label %if.then8, label %if.end10

for.body:                                         ; preds = %for.body.preheader, %for.inc
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.inc ]
  %TotalAFComponents.0152 = phi i32 [ 0, %for.body.preheader ], [ %TotalAFComponents.1, %for.inc ]
  %arrayidx = getelementptr inbounds ptr, ptr %AFItemWRTOperands, i64 %indvars.iv
  %1 = load ptr, ptr %arrayidx, align 8, !tbaa !8
  %cmp1.not = icmp eq ptr %1, null
  br i1 %cmp1.not, label %for.inc, label %if.then

if.then:                                          ; preds = %for.body
  %2 = load ptr, ptr %1, align 8, !tbaa !8
  %NumAFComponents = getelementptr inbounds %struct.AFItem, ptr %2, i64 0, i32 2
  %3 = load i32, ptr %NumAFComponents, align 8, !tbaa !46
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %.pn = phi i32 [ %3, %if.then ], [ 1, %for.body ]
  %TotalAFComponents.1 = add nsw i32 %.pn, %TotalAFComponents.0152
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body, !llvm.loop !55

if.then8:                                         ; preds = %fAFCreateAFItem.exit
  %call9 = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.65)
  call void @exit(i32 noundef 1) #23
  unreachable

if.end10:                                         ; preds = %fAFCreateAFItem.exit
  %inc11 = add nsw i32 %0, 1
  store i32 %inc11, ptr @AFItemCounter, align 4, !tbaa !16
  %cmp14163 = icmp sgt i32 %NumOperands, 0
  br i1 %cmp14163, label %for.body17.preheader, label %for.cond.cleanup16

for.body17.preheader:                             ; preds = %if.end10
  %AFComponentCounter.promoted159 = load i32, ptr @AFComponentCounter, align 4, !tbaa !16
  %wide.trip.count173 = zext i32 %NumOperands to i64
  br label %for.body17

for.cond.cleanup16:                               ; preds = %for.inc90, %if.end10
  %4 = load ptr, ptr @AFs, align 8, !tbaa !8
  %AFItems = getelementptr inbounds %struct.AFTable, ptr %4, i64 0, i32 1
  %5 = load ptr, ptr %AFItems, align 8, !tbaa !45
  %6 = load i64, ptr %4, align 8, !tbaa !43
  %arrayidx93 = getelementptr inbounds ptr, ptr %5, i64 %6
  store ptr %call.i, ptr %arrayidx93, align 8, !tbaa !8
  %7 = load ptr, ptr @AFs, align 8, !tbaa !8
  %8 = load i64, ptr %7, align 8, !tbaa !43
  %inc95 = add i64 %8, 1
  store i64 %inc95, ptr %7, align 8, !tbaa !43
  %AFItems96 = getelementptr inbounds %struct.AFTable, ptr %7, i64 0, i32 1
  %9 = load ptr, ptr %AFItems96, align 8, !tbaa !45
  %arrayidx98 = getelementptr inbounds ptr, ptr %9, i64 %8
  ret ptr %arrayidx98

for.body17:                                       ; preds = %for.body17.preheader, %for.inc90
  %indvars.iv170 = phi i64 [ 0, %for.body17.preheader ], [ %indvars.iv.next171, %for.inc90 ]
  %AFComponentCounter.promoted162164 = phi i32 [ %AFComponentCounter.promoted159, %for.body17.preheader ], [ %AFComponentCounter.promoted160, %for.inc90 ]
  %arrayidx19 = getelementptr inbounds ptr, ptr %AFItemWRTOperands, i64 %indvars.iv170
  %10 = load ptr, ptr %arrayidx19, align 8, !tbaa !8
  %cmp20.not = icmp eq ptr %10, null
  br i1 %cmp20.not, label %if.else69, label %for.cond23.preheader

for.cond23.preheader:                             ; preds = %for.body17
  %11 = load ptr, ptr %arrayidx19, align 8, !tbaa !8
  %12 = load ptr, ptr %11, align 8, !tbaa !8
  %NumAFComponents26155 = getelementptr inbounds %struct.AFItem, ptr %12, i64 0, i32 2
  %13 = load i32, ptr %NumAFComponents26155, align 8, !tbaa !46
  %cmp27156 = icmp sgt i32 %13, 0
  br i1 %cmp27156, label %for.body30, label %for.inc90

for.body30:                                       ; preds = %for.cond23.preheader, %fAFCreateAFComponent.exit
  %indvars.iv167 = phi i64 [ %indvars.iv.next168, %fAFCreateAFComponent.exit ], [ 0, %for.cond23.preheader ]
  %14 = phi ptr [ %33, %fAFCreateAFComponent.exit ], [ %12, %for.cond23.preheader ]
  %15 = phi ptr [ %32, %fAFCreateAFComponent.exit ], [ %11, %for.cond23.preheader ]
  %inc59154157 = phi i32 [ %inc59, %fAFCreateAFComponent.exit ], [ %AFComponentCounter.promoted162164, %for.cond23.preheader ]
  %call.i131 = call noalias dereferenceable_or_null(48) ptr @malloc(i64 noundef 48) #22
  %cmp.i132 = icmp eq ptr %call.i131, null
  br i1 %cmp.i132, label %if.then.i134, label %fAFCreateAFComponent.exit

if.then.i134:                                     ; preds = %for.body30
  %call1.i133 = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.59) #21
  call void @exit(i32 noundef 1) #23
  unreachable

fAFCreateAFComponent.exit:                        ; preds = %for.body30
  %Components33 = getelementptr inbounds %struct.AFItem, ptr %14, i64 0, i32 1
  %16 = load ptr, ptr %Components33, align 8, !tbaa !49
  %arrayidx35 = getelementptr inbounds ptr, ptr %16, i64 %indvars.iv167
  %17 = load ptr, ptr %arrayidx35, align 8, !tbaa !8
  %Input = getelementptr inbounds %struct.AFProduct, ptr %17, i64 0, i32 3
  %18 = load ptr, ptr %Input, align 8, !tbaa !56
  %Input36 = getelementptr inbounds %struct.AFProduct, ptr %call.i131, i64 0, i32 3
  store ptr %18, ptr %Input36, align 8, !tbaa !56
  store i32 %inc59154157, ptr %call.i131, align 8, !tbaa !50
  %19 = load ptr, ptr %AC, align 8, !tbaa !8
  %Factor = getelementptr inbounds %struct.AFProduct, ptr %call.i131, i64 0, i32 1
  store ptr %19, ptr %Factor, align 8, !tbaa !57
  %20 = load ptr, ptr %15, align 8, !tbaa !8
  %Components40 = getelementptr inbounds %struct.AFItem, ptr %20, i64 0, i32 1
  %21 = load ptr, ptr %Components40, align 8, !tbaa !49
  %arrayidx42 = getelementptr inbounds ptr, ptr %21, i64 %indvars.iv167
  %22 = load ptr, ptr %arrayidx42, align 8, !tbaa !8
  %ProductTail = getelementptr inbounds %struct.AFProduct, ptr %call.i131, i64 0, i32 2
  store ptr %22, ptr %ProductTail, align 8, !tbaa !41
  %23 = load ptr, ptr %15, align 8, !tbaa !8
  %Components45 = getelementptr inbounds %struct.AFItem, ptr %23, i64 0, i32 1
  %24 = load ptr, ptr %Components45, align 8, !tbaa !49
  %arrayidx47 = getelementptr inbounds ptr, ptr %24, i64 %indvars.iv167
  %25 = load ptr, ptr %arrayidx47, align 8, !tbaa !8
  %Height = getelementptr inbounds %struct.AFProduct, ptr %25, i64 0, i32 4
  %26 = load i32, ptr %Height, align 8, !tbaa !39
  %add48 = add nsw i32 %26, 1
  %Height49 = getelementptr inbounds %struct.AFProduct, ptr %call.i131, i64 0, i32 4
  store i32 %add48, ptr %Height49, align 8, !tbaa !39
  %AF = getelementptr inbounds %struct.AFProduct, ptr %25, i64 0, i32 5
  %27 = load double, ptr %AF, align 8, !tbaa !53
  %ACWRTOperands = getelementptr inbounds %struct.ACItem, ptr %19, i64 0, i32 6
  %28 = load ptr, ptr %ACWRTOperands, align 8, !tbaa !23
  %arrayidx56 = getelementptr inbounds double, ptr %28, i64 %indvars.iv170
  %29 = load double, ptr %arrayidx56, align 8, !tbaa !10
  %mul57 = fmul double %27, %29
  %AF58 = getelementptr inbounds %struct.AFProduct, ptr %call.i131, i64 0, i32 5
  store double %mul57, ptr %AF58, align 8, !tbaa !53
  %inc59 = add nsw i32 %inc59154157, 1
  store i32 %inc59, ptr @AFComponentCounter, align 4, !tbaa !16
  %30 = load ptr, ptr %Components, align 8, !tbaa !49
  %31 = load i32, ptr %NumAFComponents5, align 8, !tbaa !46
  %idxprom62 = sext i32 %31 to i64
  %arrayidx63 = getelementptr inbounds ptr, ptr %30, i64 %idxprom62
  store ptr %call.i131, ptr %arrayidx63, align 8, !tbaa !8
  %inc65 = add nsw i32 %31, 1
  store i32 %inc65, ptr %NumAFComponents5, align 8, !tbaa !46
  %indvars.iv.next168 = add nuw nsw i64 %indvars.iv167, 1
  %32 = load ptr, ptr %arrayidx19, align 8, !tbaa !8
  %33 = load ptr, ptr %32, align 8, !tbaa !8
  %NumAFComponents26 = getelementptr inbounds %struct.AFItem, ptr %33, i64 0, i32 2
  %34 = load i32, ptr %NumAFComponents26, align 8, !tbaa !46
  %35 = sext i32 %34 to i64
  %cmp27 = icmp slt i64 %indvars.iv.next168, %35
  br i1 %cmp27, label %for.body30, label %for.inc90, !llvm.loop !58

if.else69:                                        ; preds = %for.body17
  %call.i135 = call noalias dereferenceable_or_null(48) ptr @malloc(i64 noundef 48) #22
  %cmp.i136 = icmp eq ptr %call.i135, null
  br i1 %cmp.i136, label %if.then.i138, label %fAFCreateAFComponent.exit139

if.then.i138:                                     ; preds = %if.else69
  %call1.i137 = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.59) #21
  call void @exit(i32 noundef 1) #23
  unreachable

fAFCreateAFComponent.exit139:                     ; preds = %if.else69
  %36 = load ptr, ptr %AC, align 8, !tbaa !8
  %OperandNames = getelementptr inbounds %struct.ACItem, ptr %36, i64 0, i32 4
  %37 = load ptr, ptr %OperandNames, align 8, !tbaa !26
  %arrayidx72 = getelementptr inbounds ptr, ptr %37, i64 %indvars.iv170
  %38 = load ptr, ptr %arrayidx72, align 8, !tbaa !8
  %Input73 = getelementptr inbounds %struct.AFProduct, ptr %call.i135, i64 0, i32 3
  store ptr %38, ptr %Input73, align 8, !tbaa !56
  store i32 %AFComponentCounter.promoted162164, ptr %call.i135, align 8, !tbaa !50
  %Factor75 = getelementptr inbounds %struct.AFProduct, ptr %call.i135, i64 0, i32 1
  store ptr %36, ptr %Factor75, align 8, !tbaa !57
  %ProductTail76 = getelementptr inbounds %struct.AFProduct, ptr %call.i135, i64 0, i32 2
  store ptr null, ptr %ProductTail76, align 8, !tbaa !41
  %Height77 = getelementptr inbounds %struct.AFProduct, ptr %call.i135, i64 0, i32 4
  store i32 1, ptr %Height77, align 8, !tbaa !39
  %ACWRTOperands78 = getelementptr inbounds %struct.ACItem, ptr %36, i64 0, i32 6
  %39 = load ptr, ptr %ACWRTOperands78, align 8, !tbaa !23
  %arrayidx80 = getelementptr inbounds double, ptr %39, i64 %indvars.iv170
  %40 = load double, ptr %arrayidx80, align 8, !tbaa !10
  %AF81 = getelementptr inbounds %struct.AFProduct, ptr %call.i135, i64 0, i32 5
  store double %40, ptr %AF81, align 8, !tbaa !53
  %inc82 = add nsw i32 %AFComponentCounter.promoted162164, 1
  store i32 %inc82, ptr @AFComponentCounter, align 4, !tbaa !16
  %41 = load ptr, ptr %Components, align 8, !tbaa !49
  %42 = load i32, ptr %NumAFComponents5, align 8, !tbaa !46
  %idxprom85 = sext i32 %42 to i64
  %arrayidx86 = getelementptr inbounds ptr, ptr %41, i64 %idxprom85
  store ptr %call.i135, ptr %arrayidx86, align 8, !tbaa !8
  %inc88 = add nsw i32 %42, 1
  store i32 %inc88, ptr %NumAFComponents5, align 8, !tbaa !46
  br label %for.inc90

for.inc90:                                        ; preds = %fAFCreateAFComponent.exit, %for.cond23.preheader, %fAFCreateAFComponent.exit139
  %AFComponentCounter.promoted160 = phi i32 [ %inc82, %fAFCreateAFComponent.exit139 ], [ %AFComponentCounter.promoted162164, %for.cond23.preheader ], [ %inc59, %fAFCreateAFComponent.exit ]
  %indvars.iv.next171 = add nuw nsw i64 %indvars.iv170, 1
  %exitcond174.not = icmp eq i64 %indvars.iv.next171, %wide.trip.count173
  br i1 %exitcond174.not, label %for.cond.cleanup16, label %for.body17, !llvm.loop !59
}

; Function Attrs: nounwind uwtable
define dso_local void @fAFPrintTopAmplificationPaths() local_unnamed_addr #5 {
entry:
  %puts = call i32 @puts(ptr nonnull @str.83)
  %0 = load ptr, ptr @AFs, align 8, !tbaa !8
  %AFItems = getelementptr inbounds %struct.AFTable, ptr %0, i64 0, i32 1
  %1 = load ptr, ptr %AFItems, align 8, !tbaa !45
  %2 = load i64, ptr %0, align 8, !tbaa !43
  %sub = add i64 %2, -1
  %arrayidx = getelementptr inbounds ptr, ptr %1, i64 %sub
  %3 = load ptr, ptr %arrayidx, align 8, !tbaa !8
  %Components = getelementptr inbounds %struct.AFItem, ptr %3, i64 0, i32 1
  %4 = load ptr, ptr %Components, align 8, !tbaa !49
  %NumAFComponents = getelementptr inbounds %struct.AFItem, ptr %3, i64 0, i32 2
  %5 = load i32, ptr %NumAFComponents, align 8, !tbaa !46
  %conv = sext i32 %5 to i64
  call void @qsort(ptr noundef %4, i64 noundef %conv, i64 noundef 8, ptr noundef nonnull @fAFComparator) #21
  %putchar = call i32 @putchar(i32 10)
  %puts77 = call i32 @puts(ptr nonnull @str.88)
  %6 = load ptr, ptr @AFs, align 8, !tbaa !8
  %AFItems789 = getelementptr inbounds %struct.AFTable, ptr %6, i64 0, i32 1
  %7 = load ptr, ptr %AFItems789, align 8, !tbaa !45
  %8 = load i64, ptr %6, align 8, !tbaa !43
  %sub990 = add i64 %8, -1
  %arrayidx1091 = getelementptr inbounds ptr, ptr %7, i64 %sub990
  %9 = load ptr, ptr %arrayidx1091, align 8, !tbaa !8
  %NumAFComponents1192 = getelementptr inbounds %struct.AFItem, ptr %9, i64 0, i32 2
  %10 = load i32, ptr %NumAFComponents1192, align 8, !tbaa !46
  %cmp93 = icmp sgt i32 %10, 0
  br i1 %cmp93, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup56, %entry
  %putchar78 = call i32 @putchar(i32 10)
  %puts79 = call i32 @puts(ptr nonnull @str.89)
  ret void

for.body:                                         ; preds = %entry, %for.cond.cleanup56
  %indvars.iv96 = phi i64 [ %indvars.iv.next97, %for.cond.cleanup56 ], [ 0, %entry ]
  %11 = phi ptr [ %31, %for.cond.cleanup56 ], [ %9, %entry ]
  %Components18 = getelementptr inbounds %struct.AFItem, ptr %11, i64 0, i32 1
  %12 = load ptr, ptr %Components18, align 8, !tbaa !49
  %arrayidx19 = getelementptr inbounds ptr, ptr %12, i64 %indvars.iv96
  %13 = load ptr, ptr %arrayidx19, align 8, !tbaa !8
  %Height.i = getelementptr inbounds %struct.AFProduct, ptr %13, i64 0, i32 4
  %14 = load i32, ptr %Height.i, align 8, !tbaa !39
  %conv.i = sext i32 %14 to i64
  %mul.i = shl nsw i64 %conv.i, 3
  %call.i = call noalias ptr @malloc(i64 noundef %mul.i) #22
  %cmp.i = icmp eq ptr %call.i, null
  br i1 %cmp.i, label %if.then.i, label %if.end.i

if.then.i:                                        ; preds = %for.body
  %call2.i = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.61) #21
  call void @exit(i32 noundef 1) #23
  unreachable

if.end.i:                                         ; preds = %for.body
  store ptr %13, ptr %call.i, align 8, !tbaa !8
  %cmp416.i = icmp sgt i32 %14, 1
  br i1 %cmp416.i, label %for.body.preheader.i, label %fAFFlattenAFComponentsPath.exit

for.body.preheader.i:                             ; preds = %if.end.i
  %wide.trip.count.i = zext i32 %14 to i64
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %for.body.preheader.i
  %indvars.iv.i = phi i64 [ 1, %for.body.preheader.i ], [ %indvars.iv.next.i, %for.body.i ]
  %ProductObjectWalker.017.i = phi ptr [ %13, %for.body.preheader.i ], [ %15, %for.body.i ]
  %ProductTail.i = getelementptr inbounds %struct.AFProduct, ptr %ProductObjectWalker.017.i, i64 0, i32 2
  %15 = load ptr, ptr %ProductTail.i, align 8, !tbaa !41
  %arrayidx6.i = getelementptr inbounds ptr, ptr %call.i, i64 %indvars.iv.i
  store ptr %15, ptr %arrayidx6.i, align 8, !tbaa !8
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
  %exitcond.not.i = icmp eq i64 %indvars.iv.next.i, %wide.trip.count.i
  br i1 %exitcond.not.i, label %fAFFlattenAFComponentsPath.exit, label %for.body.i, !llvm.loop !42

fAFFlattenAFComponentsPath.exit:                  ; preds = %for.body.i, %if.end.i
  %AF = getelementptr inbounds %struct.AFProduct, ptr %13, i64 0, i32 5
  %16 = load double, ptr %AF, align 8, !tbaa !53
  %17 = load i32, ptr %13, align 8, !tbaa !50
  %Input = getelementptr inbounds %struct.AFProduct, ptr %13, i64 0, i32 3
  %18 = load ptr, ptr %Input, align 8, !tbaa !56
  %call42 = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.69, double noundef %16, i32 noundef %17, ptr noundef %18)
  %19 = load ptr, ptr %call.i, align 8, !tbaa !8
  %20 = load i32, ptr %19, align 8, !tbaa !50
  %call45 = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.29, i32 noundef %20)
  %21 = load ptr, ptr @AFs, align 8, !tbaa !8
  %AFItems4781 = getelementptr inbounds %struct.AFTable, ptr %21, i64 0, i32 1
  %22 = load ptr, ptr %AFItems4781, align 8, !tbaa !45
  %23 = load i64, ptr %21, align 8, !tbaa !43
  %sub4982 = add i64 %23, -1
  %arrayidx5083 = getelementptr inbounds ptr, ptr %22, i64 %sub4982
  %24 = load ptr, ptr %arrayidx5083, align 8, !tbaa !8
  %Components5184 = getelementptr inbounds %struct.AFItem, ptr %24, i64 0, i32 1
  %25 = load ptr, ptr %Components5184, align 8, !tbaa !49
  %arrayidx5385 = getelementptr inbounds ptr, ptr %25, i64 %indvars.iv96
  %26 = load ptr, ptr %arrayidx5385, align 8, !tbaa !8
  %Height86 = getelementptr inbounds %struct.AFProduct, ptr %26, i64 0, i32 4
  %27 = load i32, ptr %Height86, align 8, !tbaa !39
  %cmp5487 = icmp sgt i32 %27, 1
  br i1 %cmp5487, label %for.body57, label %for.cond.cleanup56

for.cond.cleanup56:                               ; preds = %for.body57, %fAFFlattenAFComponentsPath.exit
  %puts80 = call i32 @puts(ptr nonnull @str.90)
  %indvars.iv.next97 = add nuw nsw i64 %indvars.iv96, 1
  %28 = load ptr, ptr @AFs, align 8, !tbaa !8
  %AFItems7 = getelementptr inbounds %struct.AFTable, ptr %28, i64 0, i32 1
  %29 = load ptr, ptr %AFItems7, align 8, !tbaa !45
  %30 = load i64, ptr %28, align 8, !tbaa !43
  %sub9 = add i64 %30, -1
  %arrayidx10 = getelementptr inbounds ptr, ptr %29, i64 %sub9
  %31 = load ptr, ptr %arrayidx10, align 8, !tbaa !8
  %NumAFComponents11 = getelementptr inbounds %struct.AFItem, ptr %31, i64 0, i32 2
  %32 = load i32, ptr %NumAFComponents11, align 8, !tbaa !46
  %33 = call i32 @llvm.smin.i32(i32 %32, i32 5) #21
  %34 = sext i32 %33 to i64
  %cmp = icmp slt i64 %indvars.iv.next97, %34
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !llvm.loop !60

for.body57:                                       ; preds = %fAFFlattenAFComponentsPath.exit, %for.body57
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body57 ], [ 1, %fAFFlattenAFComponentsPath.exit ]
  %arrayidx59 = getelementptr inbounds ptr, ptr %call.i, i64 %indvars.iv
  %35 = load ptr, ptr %arrayidx59, align 8, !tbaa !8
  %36 = load i32, ptr %35, align 8, !tbaa !50
  %call61 = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.70, i32 noundef %36)
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %37 = load ptr, ptr @AFs, align 8, !tbaa !8
  %AFItems47 = getelementptr inbounds %struct.AFTable, ptr %37, i64 0, i32 1
  %38 = load ptr, ptr %AFItems47, align 8, !tbaa !45
  %39 = load i64, ptr %37, align 8, !tbaa !43
  %sub49 = add i64 %39, -1
  %arrayidx50 = getelementptr inbounds ptr, ptr %38, i64 %sub49
  %40 = load ptr, ptr %arrayidx50, align 8, !tbaa !8
  %Components51 = getelementptr inbounds %struct.AFItem, ptr %40, i64 0, i32 1
  %41 = load ptr, ptr %Components51, align 8, !tbaa !49
  %arrayidx53 = getelementptr inbounds ptr, ptr %41, i64 %indvars.iv96
  %42 = load ptr, ptr %arrayidx53, align 8, !tbaa !8
  %Height = getelementptr inbounds %struct.AFProduct, ptr %42, i64 0, i32 4
  %43 = load i32, ptr %Height, align 8, !tbaa !39
  %44 = sext i32 %43 to i64
  %cmp54 = icmp slt i64 %indvars.iv.next, %44
  br i1 %cmp54, label %for.body57, label %for.cond.cleanup56, !llvm.loop !61
}

; Function Attrs: nofree
declare void @qsort(ptr noundef, i64 noundef, i64 noundef, ptr nocapture noundef) local_unnamed_addr #17

; Function Attrs: nounwind uwtable
define dso_local void @fAFPrintTopFromAllAmplificationPaths() local_unnamed_addr #5 {
entry:
  %puts = call i32 @puts(ptr nonnull @str.87)
  %call1 = call ptr @fAFFlattenAllComponentPaths()
  %0 = load ptr, ptr @Paths, align 8, !tbaa !8
  %1 = load i32, ptr @AFComponentCounter, align 4, !tbaa !16
  %conv = sext i32 %1 to i64
  call void @qsort(ptr noundef %0, i64 noundef %conv, i64 noundef 8, ptr noundef nonnull @fAFComparator) #21
  %putchar = call i32 @putchar(i32 10)
  %puts43 = call i32 @puts(ptr nonnull @str.88)
  %2 = load i32, ptr @AFComponentCounter, align 4, !tbaa !16
  %cmp51 = icmp sgt i32 %2, 0
  br i1 %cmp51, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup22, %entry
  %putchar44 = call i32 @putchar(i32 10)
  %puts45 = call i32 @puts(ptr nonnull @str.89)
  ret void

for.body:                                         ; preds = %entry, %for.cond.cleanup22
  %indvars.iv54 = phi i64 [ %indvars.iv.next55, %for.cond.cleanup22 ], [ 0, %entry ]
  %3 = load ptr, ptr @Paths, align 8, !tbaa !8
  %arrayidx = getelementptr inbounds ptr, ptr %3, i64 %indvars.iv54
  %4 = load ptr, ptr %arrayidx, align 8, !tbaa !8
  %Height.i = getelementptr inbounds %struct.AFProduct, ptr %4, i64 0, i32 4
  %5 = load i32, ptr %Height.i, align 8, !tbaa !39
  %conv.i = sext i32 %5 to i64
  %mul.i = shl nsw i64 %conv.i, 3
  %call.i = call noalias ptr @malloc(i64 noundef %mul.i) #22
  %cmp.i = icmp eq ptr %call.i, null
  br i1 %cmp.i, label %if.then.i, label %if.end.i

if.then.i:                                        ; preds = %for.body
  %call2.i = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.61) #21
  call void @exit(i32 noundef 1) #23
  unreachable

if.end.i:                                         ; preds = %for.body
  store ptr %4, ptr %call.i, align 8, !tbaa !8
  %cmp416.i = icmp sgt i32 %5, 1
  br i1 %cmp416.i, label %for.body.preheader.i, label %fAFFlattenAFComponentsPath.exit

for.body.preheader.i:                             ; preds = %if.end.i
  %wide.trip.count.i = zext i32 %5 to i64
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %for.body.preheader.i
  %indvars.iv.i = phi i64 [ 1, %for.body.preheader.i ], [ %indvars.iv.next.i, %for.body.i ]
  %ProductObjectWalker.017.i = phi ptr [ %4, %for.body.preheader.i ], [ %6, %for.body.i ]
  %ProductTail.i = getelementptr inbounds %struct.AFProduct, ptr %ProductObjectWalker.017.i, i64 0, i32 2
  %6 = load ptr, ptr %ProductTail.i, align 8, !tbaa !41
  %arrayidx6.i = getelementptr inbounds ptr, ptr %call.i, i64 %indvars.iv.i
  store ptr %6, ptr %arrayidx6.i, align 8, !tbaa !8
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
  %exitcond.not.i = icmp eq i64 %indvars.iv.next.i, %wide.trip.count.i
  br i1 %exitcond.not.i, label %fAFFlattenAFComponentsPath.exit, label %for.body.i, !llvm.loop !42

fAFFlattenAFComponentsPath.exit:                  ; preds = %for.body.i, %if.end.i
  %AF = getelementptr inbounds %struct.AFProduct, ptr %4, i64 0, i32 5
  %7 = load double, ptr %AF, align 8, !tbaa !53
  %8 = load i32, ptr %4, align 8, !tbaa !50
  %Input = getelementptr inbounds %struct.AFProduct, ptr %4, i64 0, i32 3
  %9 = load ptr, ptr %Input, align 8, !tbaa !56
  %call13 = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.69, double noundef %7, i32 noundef %8, ptr noundef %9)
  %10 = load ptr, ptr %call.i, align 8, !tbaa !8
  %11 = load i32, ptr %10, align 8, !tbaa !50
  %call16 = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.29, i32 noundef %11)
  %12 = load ptr, ptr @Paths, align 8, !tbaa !8
  %arrayidx1947 = getelementptr inbounds ptr, ptr %12, i64 %indvars.iv54
  %13 = load ptr, ptr %arrayidx1947, align 8, !tbaa !8
  %Height48 = getelementptr inbounds %struct.AFProduct, ptr %13, i64 0, i32 4
  %14 = load i32, ptr %Height48, align 8, !tbaa !39
  %cmp2049 = icmp sgt i32 %14, 1
  br i1 %cmp2049, label %for.body23, label %for.cond.cleanup22

for.cond.cleanup22:                               ; preds = %for.body23, %fAFFlattenAFComponentsPath.exit
  %puts46 = call i32 @puts(ptr nonnull @str.90)
  %indvars.iv.next55 = add nuw nsw i64 %indvars.iv54, 1
  %15 = load i32, ptr @AFComponentCounter, align 4, !tbaa !16
  %16 = call i32 @llvm.smin.i32(i32 %15, i32 20) #21
  %17 = sext i32 %16 to i64
  %cmp = icmp slt i64 %indvars.iv.next55, %17
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !llvm.loop !62

for.body23:                                       ; preds = %fAFFlattenAFComponentsPath.exit, %for.body23
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body23 ], [ 1, %fAFFlattenAFComponentsPath.exit ]
  %arrayidx25 = getelementptr inbounds ptr, ptr %call.i, i64 %indvars.iv
  %18 = load ptr, ptr %arrayidx25, align 8, !tbaa !8
  %19 = load i32, ptr %18, align 8, !tbaa !50
  %call27 = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.70, i32 noundef %19)
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %20 = load ptr, ptr @Paths, align 8, !tbaa !8
  %arrayidx19 = getelementptr inbounds ptr, ptr %20, i64 %indvars.iv54
  %21 = load ptr, ptr %arrayidx19, align 8, !tbaa !8
  %Height = getelementptr inbounds %struct.AFProduct, ptr %21, i64 0, i32 4
  %22 = load i32, ptr %Height, align 8, !tbaa !39
  %23 = sext i32 %22 to i64
  %cmp20 = icmp slt i64 %indvars.iv.next, %23
  br i1 %cmp20, label %for.body23, label %for.cond.cleanup22, !llvm.loop !63
}

; Function Attrs: nounwind uwtable
define dso_local void @fAFStoreAFs() local_unnamed_addr #5 {
entry:
  %File = alloca [5000 x i8], align 16
  %puts = call i32 @puts(ptr nonnull @str.91)
  call void @llvm.lifetime.start.p0(i64 5000, ptr nonnull %File) #21
  call void @fAFGenerateFileString(ptr noundef nonnull %File, ptr noundef nonnull @.str.75, ptr noundef nonnull @.str.44)
  %call2 = call noalias ptr @fopen(ptr noundef nonnull %File, ptr noundef nonnull @.str.45)
  %0 = call i64 @fwrite(ptr nonnull @.str.46, i64 2, i64 1, ptr %call2)
  %1 = call i64 @fwrite(ptr nonnull @.str.76, i64 10, i64 1, ptr %call2)
  %2 = load ptr, ptr @AFs, align 8, !tbaa !8
  %3 = load i64, ptr %2, align 8, !tbaa !43
  %cmp159.not = icmp eq i64 %3, 0
  br i1 %cmp159.not, label %while.end, label %for.cond.preheader

for.cond.preheader:                               ; preds = %entry, %for.cond.cleanup
  %indvars.iv168 = phi i64 [ %indvars.iv.next169, %for.cond.cleanup ], [ 0, %entry ]
  %4 = load ptr, ptr @AFs, align 8, !tbaa !8
  %AFItems152 = getelementptr inbounds %struct.AFTable, ptr %4, i64 0, i32 1
  %5 = load ptr, ptr %AFItems152, align 8, !tbaa !45
  %arrayidx153 = getelementptr inbounds ptr, ptr %5, i64 %indvars.iv168
  %6 = load ptr, ptr %arrayidx153, align 8, !tbaa !8
  %NumAFComponents154 = getelementptr inbounds %struct.AFItem, ptr %6, i64 0, i32 2
  %7 = load i32, ptr %NumAFComponents154, align 8, !tbaa !46
  %cmp6155 = icmp sgt i32 %7, 0
  br i1 %cmp6155, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %if.end97, %for.cond.preheader
  %indvars.iv.next169 = add nuw i64 %indvars.iv168, 1
  %8 = load ptr, ptr @AFs, align 8, !tbaa !8
  %9 = load i64, ptr %8, align 8, !tbaa !43
  %cmp = icmp ugt i64 %9, %indvars.iv.next169
  br i1 %cmp, label %for.cond.preheader, label %while.end, !llvm.loop !64

for.body:                                         ; preds = %for.cond.preheader, %if.end97
  %indvars.iv165 = phi i64 [ %indvars.iv.next166, %if.end97 ], [ 0, %for.cond.preheader ]
  %10 = phi ptr [ %47, %if.end97 ], [ %6, %for.cond.preheader ]
  %Components = getelementptr inbounds %struct.AFItem, ptr %10, i64 0, i32 1
  %11 = load ptr, ptr %Components, align 8, !tbaa !49
  %arrayidx12 = getelementptr inbounds ptr, ptr %11, i64 %indvars.iv165
  %12 = load ptr, ptr %arrayidx12, align 8, !tbaa !8
  %Height.i = getelementptr inbounds %struct.AFProduct, ptr %12, i64 0, i32 4
  %13 = load i32, ptr %Height.i, align 8, !tbaa !39
  %conv.i = sext i32 %13 to i64
  %mul.i = shl nsw i64 %conv.i, 3
  %call.i = call noalias ptr @malloc(i64 noundef %mul.i) #22
  %cmp.i = icmp eq ptr %call.i, null
  br i1 %cmp.i, label %if.then.i, label %if.end.i

if.then.i:                                        ; preds = %for.body
  %call2.i = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.61) #21
  call void @exit(i32 noundef 1) #23
  unreachable

if.end.i:                                         ; preds = %for.body
  store ptr %12, ptr %call.i, align 8, !tbaa !8
  %cmp416.i = icmp sgt i32 %13, 1
  br i1 %cmp416.i, label %for.body.preheader.i, label %fAFFlattenAFComponentsPath.exit

for.body.preheader.i:                             ; preds = %if.end.i
  %wide.trip.count.i = zext i32 %13 to i64
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %for.body.preheader.i
  %indvars.iv.i = phi i64 [ 1, %for.body.preheader.i ], [ %indvars.iv.next.i, %for.body.i ]
  %ProductObjectWalker.017.i = phi ptr [ %12, %for.body.preheader.i ], [ %14, %for.body.i ]
  %ProductTail.i = getelementptr inbounds %struct.AFProduct, ptr %ProductObjectWalker.017.i, i64 0, i32 2
  %14 = load ptr, ptr %ProductTail.i, align 8, !tbaa !41
  %arrayidx6.i = getelementptr inbounds ptr, ptr %call.i, i64 %indvars.iv.i
  store ptr %14, ptr %arrayidx6.i, align 8, !tbaa !8
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
  %exitcond.not.i = icmp eq i64 %indvars.iv.next.i, %wide.trip.count.i
  br i1 %exitcond.not.i, label %fAFFlattenAFComponentsPath.exit, label %for.body.i, !llvm.loop !42

fAFFlattenAFComponentsPath.exit:                  ; preds = %for.body.i, %if.end.i
  %15 = load i32, ptr %12, align 8, !tbaa !50
  %Factor = getelementptr inbounds %struct.AFProduct, ptr %12, i64 0, i32 1
  %16 = load ptr, ptr %Factor, align 8, !tbaa !57
  %17 = load i32, ptr %16, align 8, !tbaa !18
  %ResultVar = getelementptr inbounds %struct.ACItem, ptr %16, i64 0, i32 3
  %18 = load ptr, ptr %ResultVar, align 8, !tbaa !22
  %ProductTail = getelementptr inbounds %struct.AFProduct, ptr %12, i64 0, i32 2
  %19 = load ptr, ptr %ProductTail, align 8, !tbaa !41
  %cmp40.not = icmp eq ptr %19, null
  br i1 %cmp40.not, label %cond.end, label %cond.true

cond.true:                                        ; preds = %fAFFlattenAFComponentsPath.exit
  %20 = load i32, ptr %19, align 8, !tbaa !50
  br label %cond.end

cond.end:                                         ; preds = %fAFFlattenAFComponentsPath.exit, %cond.true
  %cond = phi i32 [ %20, %cond.true ], [ -1, %fAFFlattenAFComponentsPath.exit ]
  %Input = getelementptr inbounds %struct.AFProduct, ptr %12, i64 0, i32 3
  %21 = load ptr, ptr %Input, align 8, !tbaa !56
  %AF = getelementptr inbounds %struct.AFProduct, ptr %12, i64 0, i32 5
  %22 = load double, ptr %AF, align 8, !tbaa !53
  %call62 = call i32 (ptr, ptr, ...) @fprintf(ptr noundef %call2, ptr noundef nonnull @.str.77, i32 noundef %15, i32 noundef %17, ptr noundef %18, i32 noundef %cond, ptr noundef %21, double noundef %22)
  %cmp63 = icmp sgt i32 %call62, 0
  br i1 %cmp63, label %if.then, label %if.end97

if.then:                                          ; preds = %cond.end
  %23 = load ptr, ptr %call.i, align 8, !tbaa !8
  %24 = load i32, ptr %23, align 8, !tbaa !50
  %call67 = call i32 (ptr, ptr, ...) @fprintf(ptr noundef %call2, ptr noundef nonnull @.str.78, i32 noundef %24)
  %25 = load ptr, ptr @AFs, align 8, !tbaa !8
  %AFItems69144 = getelementptr inbounds %struct.AFTable, ptr %25, i64 0, i32 1
  %26 = load ptr, ptr %AFItems69144, align 8, !tbaa !45
  %arrayidx71145 = getelementptr inbounds ptr, ptr %26, i64 %indvars.iv168
  %27 = load ptr, ptr %arrayidx71145, align 8, !tbaa !8
  %Components72146 = getelementptr inbounds %struct.AFItem, ptr %27, i64 0, i32 1
  %28 = load ptr, ptr %Components72146, align 8, !tbaa !49
  %arrayidx74147 = getelementptr inbounds ptr, ptr %28, i64 %indvars.iv165
  %29 = load ptr, ptr %arrayidx74147, align 8, !tbaa !8
  %Height148 = getelementptr inbounds %struct.AFProduct, ptr %29, i64 0, i32 4
  %30 = load i32, ptr %Height148, align 8, !tbaa !39
  %cmp75149 = icmp sgt i32 %30, 1
  br i1 %cmp75149, label %for.body78, label %for.cond.cleanup77

for.cond.cleanup77:                               ; preds = %for.body78, %if.then
  %.lcssa143 = phi ptr [ %25, %if.then ], [ %34, %for.body78 ]
  %.lcssa = phi ptr [ %27, %if.then ], [ %36, %for.body78 ]
  %31 = load i64, ptr %.lcssa143, align 8, !tbaa !43
  %sub = add i64 %31, -1
  %cmp85 = icmp eq i64 %sub, %indvars.iv168
  br i1 %cmp85, label %land.lhs.true, label %if.else

for.body78:                                       ; preds = %if.then, %for.body78
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body78 ], [ 1, %if.then ]
  %arrayidx80 = getelementptr inbounds ptr, ptr %call.i, i64 %indvars.iv
  %32 = load ptr, ptr %arrayidx80, align 8, !tbaa !8
  %33 = load i32, ptr %32, align 8, !tbaa !50
  %call82 = call i32 (ptr, ptr, ...) @fprintf(ptr noundef %call2, ptr noundef nonnull @.str.70, i32 noundef %33)
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %34 = load ptr, ptr @AFs, align 8, !tbaa !8
  %AFItems69 = getelementptr inbounds %struct.AFTable, ptr %34, i64 0, i32 1
  %35 = load ptr, ptr %AFItems69, align 8, !tbaa !45
  %arrayidx71 = getelementptr inbounds ptr, ptr %35, i64 %indvars.iv168
  %36 = load ptr, ptr %arrayidx71, align 8, !tbaa !8
  %Components72 = getelementptr inbounds %struct.AFItem, ptr %36, i64 0, i32 1
  %37 = load ptr, ptr %Components72, align 8, !tbaa !49
  %arrayidx74 = getelementptr inbounds ptr, ptr %37, i64 %indvars.iv165
  %38 = load ptr, ptr %arrayidx74, align 8, !tbaa !8
  %Height = getelementptr inbounds %struct.AFProduct, ptr %38, i64 0, i32 4
  %39 = load i32, ptr %Height, align 8, !tbaa !39
  %40 = sext i32 %39 to i64
  %cmp75 = icmp slt i64 %indvars.iv.next, %40
  br i1 %cmp75, label %for.body78, label %for.cond.cleanup77, !llvm.loop !65

land.lhs.true:                                    ; preds = %for.cond.cleanup77
  %NumAFComponents90 = getelementptr inbounds %struct.AFItem, ptr %.lcssa, i64 0, i32 2
  %41 = load i32, ptr %NumAFComponents90, align 8, !tbaa !46
  %sub91 = add nsw i32 %41, -1
  %42 = zext i32 %sub91 to i64
  %cmp92 = icmp eq i64 %indvars.iv165, %42
  br i1 %cmp92, label %if.then94, label %if.else

if.then94:                                        ; preds = %land.lhs.true
  %43 = call i64 @fwrite(ptr nonnull @.str.79, i64 6, i64 1, ptr %call2)
  br label %if.end97

if.else:                                          ; preds = %land.lhs.true, %for.cond.cleanup77
  %44 = call i64 @fwrite(ptr nonnull @.str.80, i64 7, i64 1, ptr %call2)
  br label %if.end97

if.end97:                                         ; preds = %if.then94, %if.else, %cond.end
  %indvars.iv.next166 = add nuw nsw i64 %indvars.iv165, 1
  %45 = load ptr, ptr @AFs, align 8, !tbaa !8
  %AFItems = getelementptr inbounds %struct.AFTable, ptr %45, i64 0, i32 1
  %46 = load ptr, ptr %AFItems, align 8, !tbaa !45
  %arrayidx = getelementptr inbounds ptr, ptr %46, i64 %indvars.iv168
  %47 = load ptr, ptr %arrayidx, align 8, !tbaa !8
  %NumAFComponents = getelementptr inbounds %struct.AFItem, ptr %47, i64 0, i32 2
  %48 = load i32, ptr %NumAFComponents, align 8, !tbaa !46
  %49 = sext i32 %48 to i64
  %cmp6 = icmp slt i64 %indvars.iv.next166, %49
  br i1 %cmp6, label %for.body, label %for.cond.cleanup, !llvm.loop !66

while.end:                                        ; preds = %for.cond.cleanup, %entry
  %50 = call i64 @fwrite(ptr nonnull @.str.56, i64 3, i64 1, ptr %call2)
  %51 = call i64 @fwrite(ptr nonnull @.str.57, i64 2, i64 1, ptr %call2)
  %call104 = call i32 @fclose(ptr noundef %call2)
  %call106 = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.81, ptr noundef nonnull %File)
  call void @llvm.lifetime.end.p0(i64 5000, ptr nonnull %File) #21
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind readonly willreturn uwtable
; CHECK-LABEL: @main
define dso_local i32 @main() local_unnamed_addr #16 {
entry:
; CHECK: call void @fACCreate()
; CHECK-NEXT: call void @fAFInitialize()
  %0 = load i32, ptr @a, align 4, !tbaa !16
  %cmp = icmp sgt i32 %0, 0
  %cond = select i1 %cmp, i32 123, i32 321
; CHECK: %1 = select i1 %cmp, ptr null, ptr null
; CHECK-NEXT: call void @fACStoreACs()
; CHECK-NEXT: call void @fAFStoreAFs()
; CHECK-NEXT: call void @fAFPrintTopFromAllAmplificationPaths()
  ret i32 %cond
}

; Function Attrs: argmemonly nofree nounwind willreturn
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #18

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #19

; Function Attrs: nofree nounwind
declare noundef i64 @fwrite(ptr nocapture noundef, i64 noundef, i64 noundef, ptr nocapture noundef) local_unnamed_addr #19

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare i32 @llvm.smin.i32(i32, i32) #20

; Function Attrs: nofree nounwind
declare noundef i32 @putchar(i32 noundef) local_unnamed_addr #19

attributes #0 = { mustprogress nofree norecurse nosync nounwind readnone willreturn uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { argmemonly mustprogress nocallback nofree nosync nounwind willreturn }
attributes #3 = { nofree nounwind "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { argmemonly mustprogress nofree nounwind willreturn "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { inaccessiblememonly mustprogress nofree nounwind willreturn allocsize(0) "alloc-family"="malloc" "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #7 = { noreturn nounwind "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #8 = { argmemonly mustprogress nofree nounwind readonly willreturn "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #9 = { nounwind "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #10 = { argmemonly mustprogress nofree norecurse nosync nounwind readonly willreturn uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #11 = { mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn }
attributes #12 = { mustprogress nofree nounwind willreturn writeonly "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #13 = { noinline nounwind optnone uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #14 = { mustprogress nofree nosync nounwind readnone willreturn uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #15 = { argmemonly mustprogress nofree nounwind readonly willreturn uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #16 = { mustprogress nofree norecurse nosync nounwind readonly willreturn uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #17 = { nofree "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #18 = { argmemonly nofree nounwind willreturn }
attributes #19 = { nofree nounwind }
attributes #20 = { nocallback nofree nosync nounwind readnone speculatable willreturn }
attributes #21 = { nounwind }
attributes #22 = { nounwind allocsize(0) }
attributes #23 = { noreturn nounwind }
attributes #24 = { nounwind readonly willreturn }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"clang version 15.0.0 (https://github.com/tanmaytirpankar/llvm-project.git bd4c8cb25d0c0d2ebbb3dafba13318898f1342b8)"}
!5 = !{!6, !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = !{!9, !9, i64 0}
!9 = !{!"any pointer", !6, i64 0}
!10 = !{!11, !11, i64 0}
!11 = !{!"double", !6, i64 0}
!12 = !{!13, !9, i64 8}
!13 = !{!"ACTable", !14, i64 0, !9, i64 8}
!14 = !{!"long", !6, i64 0}
!15 = !{!13, !14, i64 0}
!16 = !{!17, !17, i64 0}
!17 = !{!"int", !6, i64 0}
!18 = !{!19, !17, i64 0}
!19 = !{!"ACItem", !17, i64 0, !6, i64 4, !17, i64 8, !9, i64 16, !9, i64 24, !9, i64 32, !9, i64 40, !9, i64 48, !9, i64 56, !17, i64 64}
!20 = !{!19, !6, i64 4}
!21 = !{!19, !17, i64 8}
!22 = !{!19, !9, i64 16}
!23 = !{!19, !9, i64 40}
!24 = !{!19, !9, i64 56}
!25 = !{!19, !17, i64 64}
!26 = !{!19, !9, i64 24}
!27 = !{!19, !9, i64 32}
!28 = !{!19, !9, i64 48}
!29 = distinct !{!29, !30, !31}
!30 = !{!"llvm.loop.mustprogress"}
!31 = !{!"llvm.loop.unroll.disable"}
!32 = distinct !{!32, !30, !31}
!33 = distinct !{!33, !30, !31}
!34 = distinct !{!34, !30, !31}
!35 = distinct !{!35, !30, !31}
!36 = distinct !{!36, !30, !31}
!37 = !{!38, !38, i64 0}
!38 = !{!"float", !6, i64 0}
!39 = !{!40, !17, i64 32}
!40 = !{!"AFProduct", !17, i64 0, !9, i64 8, !9, i64 16, !9, i64 24, !17, i64 32, !11, i64 40}
!41 = !{!40, !9, i64 16}
!42 = distinct !{!42, !30, !31}
!43 = !{!44, !14, i64 0}
!44 = !{!"AFTable", !14, i64 0, !9, i64 8}
!45 = !{!44, !9, i64 8}
!46 = !{!47, !17, i64 16}
!47 = !{!"AFItem", !17, i64 0, !9, i64 8, !17, i64 16}
!48 = distinct !{!48, !30, !31}
!49 = !{!47, !9, i64 8}
!50 = !{!40, !17, i64 0}
!51 = distinct !{!51, !30, !31}
!52 = distinct !{!52, !30, !31}
!53 = !{!40, !11, i64 40}
!54 = !{!47, !17, i64 0}
!55 = distinct !{!55, !30, !31}
!56 = !{!40, !9, i64 24}
!57 = !{!40, !9, i64 8}
!58 = distinct !{!58, !30, !31}
!59 = distinct !{!59, !30, !31}
!60 = distinct !{!60, !30, !31}
!61 = distinct !{!61, !30, !31}
!62 = distinct !{!62, !30, !31}
!63 = distinct !{!63, !30, !31}
!64 = distinct !{!64, !30, !31}
!65 = distinct !{!65, !30, !31}
!66 = distinct !{!66, !30, !31}
