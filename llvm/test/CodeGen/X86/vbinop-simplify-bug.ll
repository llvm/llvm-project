; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mattr=sse2 -mcpu=corei7 -o /dev/null

; Revision 199135 introduced a wrong check in method
; DAGCombiner::SimplifyVBinOp in an attempt to refactor some code
; using the new method 'BuildVectorSDNode::isConstant' when possible.
; 
; However the modified code in method SimplifyVBinOp now wrongly
; checks that the operands of a vector bin-op are both constants.
;
; With that wrong change, this test started failing because of a
; 'fatal error in the backend':
;   Cannot select: 0x2e329d0: v4i32 = BUILD_VECTOR 0x2e2ea00, 0x2e2ea00, 0x2e2ea00, 0x2e2ea00
;       0x2e2ea00: i32 = Constant<1> [ID=4]
;       0x2e2ea00: i32 = Constant<1> [ID=4]
;       0x2e2ea00: i32 = Constant<1> [ID=4]
;       0x2e2ea00: i32 = Constant<1> [ID=4]

define <8 x i32> @reduced_test_case() {
  %Shuff = shufflevector <8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <8 x i32> <i32 1, i32 3, i32 undef, i32 7, i32 9, i32 11, i32 13, i32 15>
  %B23 = sub <8 x i32> %Shuff, <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
  ret <8 x i32> %B23
}

