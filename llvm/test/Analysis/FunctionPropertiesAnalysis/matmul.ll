; RUN: opt < %s -passes='print<func-properties>' -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -passes='print<func-properties>' -disable-output 2>&1 -enable-detailed-function-properties | FileCheck %s --check-prefix=DETAILED-PROPERTIES

define i32 @main() {
; CHECK-DAG: Printing analysis results of CFA for function 'main':
; DETAILED-PROPERTIES-DAG: Printing analysis results of CFA for function 'main':

entry:
  %retval = alloca i32, align 4
  %mat1 = alloca [2 x [2 x i32]], align 16
  %mat2 = alloca [2 x [2 x i32]], align 16
  %res = alloca [2 x [2 x i32]], align 16
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  %arraydecay = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %mat1, i64 0, i64 0
  %arraydecay1 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %mat2, i64 0, i64 0
  %arraydecay2 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %res, i64 0, i64 0
  call void @multiply([2 x i32]* %arraydecay, [2 x i32]* %arraydecay1, [2 x i32]* %arraydecay2)
  ret i32 0
}
; CHECK-DAG: BasicBlockCount: 1
; CHECK-DAG: BlocksReachedFromConditionalInstruction: 0
; CHECK-DAG: Uses: 1
; CHECK-DAG: DirectCallsToDefinedFunctions: 1
; CHECK-DAG: LoadInstCount: 0
; CHECK-DAG: StoreInstCount: 1
; CHECK-DAG: MaxLoopDepth: 0
; CHECK-DAG: TopLevelLoopCount: 0

; DETAILED-PROPERTIES-DAG: BasicBlockCount: 1
; DETAILED-PROPERTIES-DAG: BlocksReachedFromConditionalInstruction: 0
; DETAILED-PROPERTIES-DAG: Uses: 1
; DETAILED-PROPERTIES-DAG: DirectCallsToDefinedFunctions: 1
; DETAILED-PROPERTIES-DAG: LoadInstCount: 0
; DETAILED-PROPERTIES-DAG: StoreInstCount: 1
; DETAILED-PROPERTIES-DAG: MaxLoopDepth: 0
; DETAILED-PROPERTIES-DAG: TopLevelLoopCount: 0
; DETAILED-PROPERTIES-DAG: BasicBlocksWithSingleSuccessor: 0
; DETAILED-PROPERTIES-DAG: BasicBlocksWithTwoSuccessors: 0
; DETAILED-PROPERTIES-DAG: BasicBlocksWithMoreThanTwoSuccessors: 0
; DETAILED-PROPERTIES-DAG: BasicBlocksWithSinglePredecessor: 0
; DETAILED-PROPERTIES-DAG: BasicBlocksWithTwoPredecessors: 0
; DETAILED-PROPERTIES-DAG: BasicBlocksWithMoreThanTwoPredecessors: 0
; DETAILED-PROPERTIES-DAG: BigBasicBlocks: 0
; DETAILED-PROPERTIES-DAG: MediumBasicBlocks: 0
; DETAILED-PROPERTIES-DAG: SmallBasicBlocks: 1
; DETAILED-PROPERTIES-DAG: CastInstructionCount: 0
; DETAILED-PROPERTIES-DAG: FloatingPointInstructionCount: 0
; DETAILED-PROPERTIES-DAG: IntegerInstructionCount: 0
; DETAILED-PROPERTIES-DAG: ConstantIntOperandCount: 14
; DETAILED-PROPERTIES-DAG: ConstantFPOperandCount: 0
; DETAILED-PROPERTIES-DAG: ConstantOperandCount: 0
; DETAILED-PROPERTIES-DAG: InstructionOperandCount: 7
; DETAILED-PROPERTIES-DAG: BasicBlockOperandCount: 0
; DETAILED-PROPERTIES-DAG: GlobalValueOperandCount: 1
; DETAILED-PROPERTIES-DAG: InlineAsmOperandCount: 0
; DETAILED-PROPERTIES-DAG: ArgumentOperandCount: 0
; DETAILED-PROPERTIES-DAG: UnknownOperandCount: 0
; DETAILED-PROPERTIES-DAG: CriticalEdgeCount: 0
; DETAILED-PROPERTIES-DAG: ControlFlowEdgeCount: 0
; DETAILED-PROPERTIES-DAG: UnconditionalBranchCount: 0
; DETAILED-PROPERTIES-DAG: DirectCallCount: 1
; DETAILED-PROPERTIES-DAG: IndirectCallCount: 0
; DETAILED-PROPERTIES-DAG: CallReturnsIntegerCount: 0
; DETAILED-PROPERTIES-DAG: CallReturnsFloatCount: 0
; DETAILED-PROPERTIES-DAG: CallReturnsPointerCount: 0
; DETAILED-PROPERTIES-DAG: CallReturnsVectorIntCount: 0
; DETAILED-PROPERTIES-DAG: CallReturnsVectorFloatCount: 0
; DETAILED-PROPERTIES-DAG: CallReturnsVectorPointerCount: 0
; DETAILED-PROPERTIES-DAG: CallWithManyArgumentsCount: 0
; DETAILED-PROPERTIES-DAG: CallWithPointerArgumentCount: 1

define void @multiply([2 x i32]* %mat1, [2 x i32]* %mat2, [2 x i32]* %res) {
; CHECK-DAG: Printing analysis results of CFA for function 'multiply':
; DETAILED-PROPERTIES-DAG: Printing analysis results of CFA for function 'multiply':
entry:
  %mat1.addr = alloca [2 x i32]*, align 8
  %mat2.addr = alloca [2 x i32]*, align 8
  %res.addr = alloca [2 x i32]*, align 8
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %k = alloca i32, align 4
  store [2 x i32]* %mat1, [2 x i32]** %mat1.addr, align 8
  store [2 x i32]* %mat2, [2 x i32]** %mat2.addr, align 8
  store [2 x i32]* %res, [2 x i32]** %res.addr, align 8
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc24, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 2
  br i1 %cmp, label %for.body, label %for.end26

for.body:                                         ; preds = %for.cond
  store i32 0, i32* %j, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc21, %for.body
  %1 = load i32, i32* %j, align 4
  %cmp2 = icmp slt i32 %1, 2
  br i1 %cmp2, label %for.body3, label %for.end23

for.body3:                                        ; preds = %for.cond1
  %2 = load [2 x i32]*, [2 x i32]** %res.addr, align 8
  %3 = load i32, i32* %i, align 4
  %idxprom = sext i32 %3 to i64
  %arrayidx = getelementptr inbounds [2 x i32], [2 x i32]* %2, i64 %idxprom
  %4 = load i32, i32* %j, align 4
  %idxprom4 = sext i32 %4 to i64
  %arrayidx5 = getelementptr inbounds [2 x i32], [2 x i32]* %arrayidx, i64 0, i64 %idxprom4
  store i32 0, i32* %arrayidx5, align 4
  store i32 0, i32* %k, align 4
  br label %for.cond6

for.cond6:                                        ; preds = %for.inc, %for.body3
  %5 = load i32, i32* %k, align 4
  %cmp7 = icmp slt i32 %5, 2
  br i1 %cmp7, label %for.body8, label %for.end

for.body8:                                        ; preds = %for.cond6
  %6 = load [2 x i32]*, [2 x i32]** %mat1.addr, align 8
  %7 = load i32, i32* %i, align 4
  %idxprom9 = sext i32 %7 to i64
  %arrayidx10 = getelementptr inbounds [2 x i32], [2 x i32]* %6, i64 %idxprom9
  %8 = load i32, i32* %k, align 4
  %idxprom11 = sext i32 %8 to i64
  %arrayidx12 = getelementptr inbounds [2 x i32], [2 x i32]* %arrayidx10, i64 0, i64 %idxprom11
  %9 = load i32, i32* %arrayidx12, align 4
  %10 = load [2 x i32]*, [2 x i32]** %mat2.addr, align 8
  %11 = load i32, i32* %k, align 4
  %idxprom13 = sext i32 %11 to i64
  %arrayidx14 = getelementptr inbounds [2 x i32], [2 x i32]* %10, i64 %idxprom13
  %12 = load i32, i32* %j, align 4
  %idxprom15 = sext i32 %12 to i64
  %arrayidx16 = getelementptr inbounds [2 x i32], [2 x i32]* %arrayidx14, i64 0, i64 %idxprom15
  %13 = load i32, i32* %arrayidx16, align 4
  %mul = mul nsw i32 %9, %13
  %14 = load [2 x i32]*, [2 x i32]** %res.addr, align 8
  %15 = load i32, i32* %i, align 4
  %idxprom17 = sext i32 %15 to i64
  %arrayidx18 = getelementptr inbounds [2 x i32], [2 x i32]* %14, i64 %idxprom17
  %16 = load i32, i32* %j, align 4
  %idxprom19 = sext i32 %16 to i64
  %arrayidx20 = getelementptr inbounds [2 x i32], [2 x i32]* %arrayidx18, i64 0, i64 %idxprom19
  %17 = load i32, i32* %arrayidx20, align 4
  %add = add nsw i32 %17, %mul
  store i32 %add, i32* %arrayidx20, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body8
  %18 = load i32, i32* %k, align 4
  %inc = add nsw i32 %18, 1
  store i32 %inc, i32* %k, align 4
  br label %for.cond6

for.end:                                          ; preds = %for.cond6
  br label %for.inc21

for.inc21:                                        ; preds = %for.end
  %19 = load i32, i32* %j, align 4
  %inc22 = add nsw i32 %19, 1
  store i32 %inc22, i32* %j, align 4
  br label %for.cond1

for.end23:                                        ; preds = %for.cond1
  br label %for.inc24

for.inc24:                                        ; preds = %for.end23
  %20 = load i32, i32* %i, align 4
  %inc25 = add nsw i32 %20, 1
  store i32 %inc25, i32* %i, align 4
  br label %for.cond

for.end26:                                        ; preds = %for.cond
  ret void
}

; CHECK-DAG: BasicBlockCount: 13
; CHECK-DAG: BlocksReachedFromConditionalInstruction: 6
; CHECK-DAG: Uses: 2
; CHECK-DAG: DirectCallsToDefinedFunctions: 0
; CHECK-DAG: LoadInstCount: 21
; CHECK-DAG: StoreInstCount: 11
; CHECK-DAG: MaxLoopDepth: 3
; CHECK-DAG: TopLevelLoopCount: 1

; DETAILED-PROPERTIES-DAG: BasicBlockCount: 13
; DETAILED-PROPERTIES-DAG: BlocksReachedFromConditionalInstruction: 6
; DETAILED-PROPERTIES-DAG: Uses: 2
; DETAILED-PROPERTIES-DAG: DirectCallsToDefinedFunctions: 0
; DETAILED-PROPERTIES-DAG: LoadInstCount: 21
; DETAILED-PROPERTIES-DAG: StoreInstCount: 11
; DETAILED-PROPERTIES-DAG: MaxLoopDepth: 3
; DETAILED-PROPERTIES-DAG: TopLevelLoopCount: 1
; DETAILED-PROPERTIES-DAG: BasicBlocksWithSingleSuccessor: 9
; DETAILED-PROPERTIES-DAG: BasicBlocksWithTwoSuccessors: 3
; DETAILED-PROPERTIES-DAG: BasicBlocksWithMoreThanTwoSuccessors: 0
; DETAILED-PROPERTIES-DAG: BasicBlocksWithSinglePredecessor: 9
; DETAILED-PROPERTIES-DAG: BasicBlocksWithTwoPredecessors: 3
; DETAILED-PROPERTIES-DAG: BasicBlocksWithMoreThanTwoPredecessors: 0
; DETAILED-PROPERTIES-DAG: BigBasicBlocks: 0
; DETAILED-PROPERTIES-DAG: MediumBasicBlocks: 11
; DETAILED-PROPERTIES-DAG: SmallBasicBlocks: 2
; DETAILED-PROPERTIES-DAG: CastInstructionCount: 8
; DETAILED-PROPERTIES-DAG: FloatingPointInstructionCount: 0
; DETAILED-PROPERTIES-DAG: IntegerInstructionCount: 33
; DETAILED-PROPERTIES-DAG: ConstantIntOperandCount: 20
; DETAILED-PROPERTIES-DAG: ConstantFPOperandCount: 0
; DETAILED-PROPERTIES-DAG: ConstantOperandCount: 0
; DETAILED-PROPERTIES-DAG: InstructionOperandCount: 73
; DETAILED-PROPERTIES-DAG: BasicBlockOperandCount: 15
; DETAILED-PROPERTIES-DAG: GlobalValueOperandCount: 0
; DETAILED-PROPERTIES-DAG: InlineAsmOperandCount: 0
; DETAILED-PROPERTIES-DAG: ArgumentOperandCount: 3
; DETAILED-PROPERTIES-DAG: UnknownOperandCount: 0
; DETAILED-PROPERTIES-DAG: DirectCallCount: 0
; DETAILED-PROPERTIES-DAG: IndirectCallCount: 0
; DETAILED-PROPERTIES-DAG: CriticalEdgeCount: 0
; DETAILED-PROPERTIES-DAG: ControlFlowEdgeCount: 15
; DETAILED-PROPERTIES-DAG: UnconditionalBranchCount: 9
; DETAILED-PROPERTIES-DAG: CallReturnsIntegerCount: 0
; DETAILED-PROPERTIES-DAG: CallReturnsFloatCount: 0
; DETAILED-PROPERTIES-DAG: CallReturnsPointerCount: 0
; DETAILED-PROPERTIES-DAG: CallReturnsVectorIntCount: 0
; DETAILED-PROPERTIES-DAG: CallReturnsVectorFloatCount: 0
; DETAILED-PROPERTIES-DAG: CallReturnsVectorPointerCount: 0
; DETAILED-PROPERTIES-DAG: CallWithManyArgumentsCount: 0
; DETAILED-PROPERTIES-DAG: CallWithPointerArgumentCount: 0

