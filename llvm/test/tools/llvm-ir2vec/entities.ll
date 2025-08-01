; RUN: llvm-ir2vec entities | FileCheck %s

CHECK: 92
CHECK-NEXT: Ret     0
CHECK-NEXT: Br      1
CHECK-NEXT: Switch  2
CHECK-NEXT: IndirectBr      3
CHECK-NEXT: Invoke  4
CHECK-NEXT: Resume  5
CHECK-NEXT: Unreachable     6
CHECK-NEXT: CleanupRet      7
CHECK-NEXT: CatchRet        8
CHECK-NEXT: CatchSwitch     9
CHECK-NEXT: CallBr  10
CHECK-NEXT: FNeg    11
CHECK-NEXT: Add     12
CHECK-NEXT: FAdd    13
CHECK-NEXT: Sub     14
CHECK-NEXT: FSub    15
CHECK-NEXT: Mul     16
CHECK-NEXT: FMul    17
CHECK-NEXT: UDiv    18
CHECK-NEXT: SDiv    19
CHECK-NEXT: FDiv    20
CHECK-NEXT: URem    21
CHECK-NEXT: SRem    22
CHECK-NEXT: FRem    23
CHECK-NEXT: Shl     24
CHECK-NEXT: LShr    25
CHECK-NEXT: AShr    26
CHECK-NEXT: And     27
CHECK-NEXT: Or      28
CHECK-NEXT: Xor     29
CHECK-NEXT: Alloca  30
CHECK-NEXT: Load    31
CHECK-NEXT: Store   32
CHECK-NEXT: GetElementPtr   33
CHECK-NEXT: Fence   34
CHECK-NEXT: AtomicCmpXchg   35
CHECK-NEXT: AtomicRMW       36
CHECK-NEXT: Trunc   37
CHECK-NEXT: ZExt    38
CHECK-NEXT: SExt    39
CHECK-NEXT: FPToUI  40
CHECK-NEXT: FPToSI  41
CHECK-NEXT: UIToFP  42
CHECK-NEXT: SIToFP  43
CHECK-NEXT: FPTrunc 44
CHECK-NEXT: FPExt   45
CHECK-NEXT: PtrToInt        46
CHECK-NEXT: IntToPtr        47
CHECK-NEXT: BitCast 48
CHECK-NEXT: AddrSpaceCast   49
CHECK-NEXT: CleanupPad      50
CHECK-NEXT: CatchPad        51
CHECK-NEXT: ICmp    52
CHECK-NEXT: FCmp    53
CHECK-NEXT: PHI     54
CHECK-NEXT: Call    55
CHECK-NEXT: Select  56
CHECK-NEXT: UserOp1 57
CHECK-NEXT: UserOp2 58
CHECK-NEXT: VAArg   59
CHECK-NEXT: ExtractElement  60
CHECK-NEXT: InsertElement   61
CHECK-NEXT: ShuffleVector   62
CHECK-NEXT: ExtractValue    63
CHECK-NEXT: InsertValue     64
CHECK-NEXT: LandingPad      65
CHECK-NEXT: Freeze  66
CHECK-NEXT: FloatTy 67
CHECK-NEXT: FloatTy 68
CHECK-NEXT: FloatTy 69
CHECK-NEXT: FloatTy 70
CHECK-NEXT: FloatTy 71
CHECK-NEXT: FloatTy 72
CHECK-NEXT: FloatTy 73
CHECK-NEXT: VoidTy  74
CHECK-NEXT: LabelTy 75
CHECK-NEXT: MetadataTy      76
CHECK-NEXT: UnknownTy       77
CHECK-NEXT: TokenTy 78
CHECK-NEXT: IntegerTy       79
CHECK-NEXT: FunctionTy      80
CHECK-NEXT: PointerTy       81
CHECK-NEXT: StructTy        82
CHECK-NEXT: ArrayTy 83
CHECK-NEXT: VectorTy        84
CHECK-NEXT: VectorTy        85
CHECK-NEXT: PointerTy       86
CHECK-NEXT: UnknownTy       87
CHECK-NEXT: Function        88
CHECK-NEXT: Pointer 89
CHECK-NEXT: Constant        90
CHECK-NEXT: Variable        91
