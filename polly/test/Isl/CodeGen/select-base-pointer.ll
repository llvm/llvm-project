; RUN: opt %loadPolly -tbaa -polly-codegen -disable-output %s
;
; Check that we do not crash here.
;
; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%"DOMParentNode" = type { %"DOMNode"*, %"DOMNode"*, %"DOMNodeListImpl" }
%"DOMDocumentRange" = type { i32 (...)** }
%"DOMXPathEvaluator" = type { i32 (...)** }
%"DOMDocumentTraversal" = type { i32 (...)** }
%"DOMNode" = type { i32 (...)** }
%"DOMNodeListImpl" = type { %"DOMNodeList", %"DOMNode"* }
%"DOMNodeList" = type { i32 (...)** }
%"DOMElementImpl" = type { %"DOMElement", %"DOMNodeImpl", %"DOMParentNode", %"DOMChildNode", %"DOMAttrMapImpl"*, %"DOMAttrMapImpl"*, i16* }
%"DOMElement" = type { %"DOMNode" }
%"DOMNodeImpl" = type <{ %"DOMNode"*, i16, [6 x i8] }>
%"DOMChildNode" = type { %"DOMNode"*, %"DOMNode"* }
%"DOMAttrMapImpl" = type <{ %"DOMNamedNodeMapImpl", i8, [7 x i8] }>
%"DOMNamedNodeMapImpl" = type { %"DOMNamedNodeMap", %"DOMNode"* }
%"DOMNamedNodeMap" = type { i32 (...)** }
%"DOMTextImpl" = type { %"DOMText", %"DOMNodeImpl", %"DOMChildNode" }
%"DOMText" = type { %"DOMCharacterData" }
%"DOMCharacterData" = type { %"DOMNode" }

; Function Attrs: uwtable
define void @_ZN11xercesc_2_513DOMParentNode9lastChildEPNS_7DOMNodeE(%"DOMParentNode"* %this, %"DOMNode"* %node) #0 align 2 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %fFirstChild = getelementptr inbounds %"DOMParentNode", %"DOMParentNode"* %this, i32 0, i32 1
  %0 = load %"DOMNode"*, %"DOMNode"** %fFirstChild, align 8, !tbaa !1
  %cmp = icmp ne %"DOMNode"* %0, null
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry.split
  %1 = bitcast %"DOMNode"* %0 to %"DOMElementImpl"*
  %fNode.i = getelementptr inbounds %"DOMElementImpl", %"DOMElementImpl"* %1, i32 0, i32 1
  %flags.i.i = getelementptr inbounds %"DOMNodeImpl", %"DOMNodeImpl"* %fNode.i, i32 0, i32 1
  %2 = load i16, i16* %flags.i.i, align 8, !tbaa !7
  %3 = bitcast %"DOMNode"* %0 to %"DOMTextImpl"*
  %fChild.i = getelementptr inbounds %"DOMTextImpl", %"DOMTextImpl"* %3, i32 0, i32 2
  %fChild1.i = getelementptr inbounds %"DOMElementImpl", %"DOMElementImpl"* %1, i32 0, i32 3
  %retval.0.i = select i1 undef, %"DOMChildNode"* %fChild.i, %"DOMChildNode"* %fChild1.i
  %previousSibling = getelementptr inbounds %"DOMChildNode", %"DOMChildNode"* %retval.0.i, i32 0, i32 0
  store %"DOMNode"* %node, %"DOMNode"** %previousSibling, align 8, !tbaa !10
  br label %if.end

if.end:                                           ; preds = %if.then, %entry.split
  ret void
}

!0 = !{!"clang version 3.9.0"}
!1 = !{!2, !3, i64 8}
!2 = !{!"_ZTSN11xercesc_2_513DOMParentNodeE", !3, i64 0, !3, i64 8, !6, i64 16}
!3 = !{!"any pointer", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = !{!"_ZTSN11xercesc_2_515DOMNodeListImplE", !3, i64 8}
!7 = !{!8, !9, i64 8}
!8 = !{!"_ZTSN11xercesc_2_511DOMNodeImplE", !3, i64 0, !9, i64 8}
!9 = !{!"short", !4, i64 0}
!10 = !{!11, !3, i64 0}
!11 = !{!"_ZTSN11xercesc_2_512DOMChildNodeE", !3, i64 0, !3, i64 8}
