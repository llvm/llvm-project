; RUN: opt -passes='function(require<domtree>),extract-blocks,function(require<domtree>)' -debug-pass-manager -disable-output %s 2>&1 | FileCheck %s
; RUN: opt -passes='function(require<domtree>),extract-blocks,function(verify<domtree>)' -disable-output %s

; The landing pad split runs even with an empty block list, so the CFG changes.

; CHECK: Running analysis: DominatorTreeAnalysis on foo
; CHECK: Running pass: BlockExtractorPass
; CHECK: Running analysis: DominatorTreeAnalysis on foo

define void @foo() personality ptr @__gxx_personality_v0 {
entry:
  invoke void @bar()
          to label %exit unwind label %lpad

lpad:
  %0 = landingpad { ptr, i32 }
          catch ptr null
  invoke void @bar()
          to label %exit unwind label %lpad2

lpad2:
  %1 = landingpad { ptr, i32 }
          cleanup
  invoke void @bar()
          to label %exit unwind label %lpad

exit:
  ret void
}

declare void @bar()

declare i32 @__gxx_personality_v0(...)
