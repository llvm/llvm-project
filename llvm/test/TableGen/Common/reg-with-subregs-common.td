include "llvm/Target/Target.td"

def TestTargetInstrInfo : InstrInfo;

def TestTarget : Target {
  let InstructionSet = TestTargetInstrInfo;
}

class Indexes<int N> {
  list<int> all = [0,   1,  2,  3,  4,  5,  6 , 7,
                   8,   9, 10, 11, 12, 13, 14, 15,
                   16, 17, 18, 19, 20, 21, 22, 23,
                   24, 25, 26, 27, 28, 29, 30, 31];
  list<int> slice =
    !foldl([]<int>, all, acc, cur,
           !listconcat(acc, !if(!lt(cur, N), [cur], [])));
}

#ifdef USE_NAMESPACE
  let Namespace = "TestNamespace" in {
#endif
foreach Index = 0...31 in {
  def sub#Index : SubRegIndex<32, !shl(Index, 5)>;
}

foreach Size = {2,4,8,16} in {
  foreach Index = Indexes<!add(33, !mul(Size, -1))>.slice in {
    def !foldl("", Indexes<Size>.slice, acc, cur,
               !strconcat(acc#!if(!eq(acc,""),"","_"), "sub"#!add(cur, Index))) :
      SubRegIndex<!mul(Size, 32), !shl(Index, 5)> {
      let CoveringSubRegIndices =
        !foldl([]<SubRegIndex>, Indexes<Size>.slice, acc, cur,
               !listconcat(acc, [!cast<SubRegIndex>(sub#!add(cur, Index))]));
    }
  }
}

foreach Index = 0...255 in {
  def R#Index : Register <"r"#Index>;
}
#ifdef USE_NAMESPACE
}
#endif

def GPR32 : RegisterClass<"TestTarget", [i32], 32,
                          (add (sequence "R%u", 0, 255))>;

def GPR64 : RegisterTuples<[sub0, sub1],
                           [(decimate (shl GPR32, 0), 1),
                            (decimate (shl GPR32, 1), 1)
                           ]>;

def GPR128 : RegisterTuples<[sub0, sub1, sub2, sub3],
                            [
                             (decimate (shl GPR32, 0), 1),
                             (decimate (shl GPR32, 1), 1),
                             (decimate (shl GPR32, 2), 1),
                             (decimate (shl GPR32, 3), 1)
                            ]>;

def GPR256 : RegisterTuples<[sub0, sub1, sub2, sub3, sub4, sub5, sub6, sub7],
                             [
                              (decimate (shl GPR32, 0), 1),
                              (decimate (shl GPR32, 1), 1),
                              (decimate (shl GPR32, 2), 1),
                              (decimate (shl GPR32, 3), 1),
                              (decimate (shl GPR32, 4), 1),
                              (decimate (shl GPR32, 5), 1),
                              (decimate (shl GPR32, 6), 1),
                              (decimate (shl GPR32, 7), 1)
                             ]>;

def GPR512 : RegisterTuples<[sub0, sub1, sub2, sub3, sub4, sub5, sub6, sub7,
                             sub8, sub9, sub10, sub11, sub12, sub13, sub14, sub15],
                             [
                              (decimate (shl GPR32, 0), 1),
                              (decimate (shl GPR32, 1), 1),
                              (decimate (shl GPR32, 2), 1),
                              (decimate (shl GPR32, 3), 1),
                              (decimate (shl GPR32, 4), 1),
                              (decimate (shl GPR32, 5), 1),
                              (decimate (shl GPR32, 6), 1),
                              (decimate (shl GPR32, 7), 1),
                              (decimate (shl GPR32, 8), 1),
                              (decimate (shl GPR32, 9), 1),
                              (decimate (shl GPR32, 10), 1),
                              (decimate (shl GPR32, 11), 1),
                              (decimate (shl GPR32, 12), 1),
                              (decimate (shl GPR32, 13), 1),
                              (decimate (shl GPR32, 14), 1),
                              (decimate (shl GPR32, 15), 1)
                             ]>;

def GPR1024 : RegisterTuples<[sub0, sub1, sub2, sub3, sub4, sub5, sub6, sub7,
                              sub8, sub9, sub10, sub11, sub12, sub13, sub14, sub15,
                              sub16, sub17, sub18, sub19, sub20, sub21, sub22, sub23,
                              sub24, sub25, sub26, sub27, sub28, sub29, sub30, sub31],
                             [
                              (decimate (shl GPR32, 0), 1),
                              (decimate (shl GPR32, 1), 1),
                              (decimate (shl GPR32, 2), 1),
                              (decimate (shl GPR32, 3), 1),
                              (decimate (shl GPR32, 4), 1),
                              (decimate (shl GPR32, 5), 1),
                              (decimate (shl GPR32, 6), 1),
                              (decimate (shl GPR32, 7), 1),
                              (decimate (shl GPR32, 8), 1),
                              (decimate (shl GPR32, 9), 1),
                              (decimate (shl GPR32, 10), 1),
                              (decimate (shl GPR32, 11), 1),
                              (decimate (shl GPR32, 12), 1),
                              (decimate (shl GPR32, 13), 1),
                              (decimate (shl GPR32, 14), 1),
                              (decimate (shl GPR32, 15), 1),
                              (decimate (shl GPR32, 16), 1),
                              (decimate (shl GPR32, 17), 1),
                              (decimate (shl GPR32, 18), 1),
                              (decimate (shl GPR32, 19), 1),
                              (decimate (shl GPR32, 20), 1),
                              (decimate (shl GPR32, 21), 1),
                              (decimate (shl GPR32, 22), 1),
                              (decimate (shl GPR32, 23), 1),
                              (decimate (shl GPR32, 24), 1),
                              (decimate (shl GPR32, 25), 1),
                              (decimate (shl GPR32, 26), 1),
                              (decimate (shl GPR32, 27), 1),
                              (decimate (shl GPR32, 28), 1),
                              (decimate (shl GPR32, 29), 1),
                              (decimate (shl GPR32, 30), 1),
                              (decimate (shl GPR32, 31), 1)
                             ]>;

#ifdef USE_NAMESPACE
  let Namespace = "TestNamespace" in {
#endif
def GPR_64 : RegisterClass<"", [v2i32], 64, (add GPR64)>;
def GPR_1024 : RegisterClass<"", [v32i32], 1024, (add GPR1024)>;
#ifdef USE_NAMESPACE
}
#endif
