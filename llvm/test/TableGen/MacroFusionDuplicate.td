// RUN: not llvm-tblgen -gen-macro-fusion-pred -DFUSION_FOR -I %p/../../include %s 2>&1 | FileCheck --check-prefix=FUSION-FOR %s
// RUN: not llvm-tblgen -gen-macro-fusion-pred -DFUSION -I %p/../../include %s 2>&1 | FileCheck --check-prefix=FUSION %s

// Check that a subtarget feature may only have one set of fusion predicates.

include "llvm/Target/Target.td"

def TestInstrInfo : InstrInfo { }

def Test : Target {
  let InstructionSet = TestInstrInfo;
}

def TestAFusion : SubtargetFeature<"fusion-a", "HasAFusion", "true",
                                   "Test A Fusion">;

def TestBFusion : Fusion<"b", "HasBFusion", "Test B Fusion", []>;

#ifdef FUSION_FOR
// FUSION-FOR: [[@LINE+3]]:5: error: duplicate macro fusion for 'TestAFusion'{{$}}
// FUSION-FOR: [[@LINE+1]]:5: note: previous definition here{{$}}
def AFusion1 : FusionFor<TestAFusion, []>;
def AFusion0 : FusionFor<TestAFusion, []>;
#endif

#ifdef FUSION
// FUSION: [[@LINE+2]]:5: error: duplicate macro fusion for 'TestBFusion'{{$}}
// FUSION: [[@LINE-11]]:5: note: previous definition here{{$}}
def BFusion : FusionFor<TestBFusion, []>;
#endif
