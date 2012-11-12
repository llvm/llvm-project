; RUN: llc -O3 < %s
; This test fails with:
; Assertion failed: (!B && "UpdateTerminators requires analyzable predecessors!"), function updateTerminator, MachineBasicBlock.cpp, line 255.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.2"

%"struct.llvm::InlineAsm::ConstraintInfo" = type { i32, i8, i8, i8, i8, %"struct.std::vector<std::basic_string<char, std::char_traits<char>, std::allocator<char> >,std::allocator<std::basic_string<char, std::char_traits<char>, std::allocator<char> > > >" }
%"struct.std::_Vector_base<llvm::InlineAsm::ConstraintInfo,std::allocator<llvm::InlineAsm::ConstraintInfo> >" = type { %"struct.std::_Vector_base<llvm::InlineAsm::ConstraintInfo,std::allocator<llvm::InlineAsm::ConstraintInfo> >::_Vector_impl" }
%"struct.std::_Vector_base<llvm::InlineAsm::ConstraintInfo,std::allocator<llvm::InlineAsm::ConstraintInfo> >::_Vector_impl" = type { %"struct.llvm::InlineAsm::ConstraintInfo"*, %"struct.llvm::InlineAsm::ConstraintInfo"*, %"struct.llvm::InlineAsm::ConstraintInfo"* }
%"struct.std::_Vector_base<std::basic_string<char, std::char_traits<char>, std::allocator<char> >,std::allocator<std::basic_string<char, std::char_traits<char>, std::allocator<char> > > >" = type { %"struct.std::_Vector_base<std::basic_string<char, std::char_traits<char>, std::allocator<char> >,std::allocator<std::basic_string<char, std::char_traits<char>, std::allocator<char> > > >::_Vector_impl" }
%"struct.std::_Vector_base<std::basic_string<char, std::char_traits<char>, std::allocator<char> >,std::allocator<std::basic_string<char, std::char_traits<char>, std::allocator<char> > > >::_Vector_impl" = type { %"struct.std::string"*, %"struct.std::string"*, %"struct.std::string"* }
%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >::_Alloc_hider" = type { i8* }
%"struct.std::string" = type { %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >::_Alloc_hider" }
%"struct.std::vector<llvm::InlineAsm::ConstraintInfo,std::allocator<llvm::InlineAsm::ConstraintInfo> >" = type { %"struct.std::_Vector_base<llvm::InlineAsm::ConstraintInfo,std::allocator<llvm::InlineAsm::ConstraintInfo> >" }
%"struct.std::vector<std::basic_string<char, std::char_traits<char>, std::allocator<char> >,std::allocator<std::basic_string<char, std::char_traits<char>, std::allocator<char> > > >" = type { %"struct.std::_Vector_base<std::basic_string<char, std::char_traits<char>, std::allocator<char> >,std::allocator<std::basic_string<char, std::char_traits<char>, std::allocator<char> > > >" }

define zeroext i8 @_ZN4llvm9InlineAsm14ConstraintInfo5ParseENS_9StringRefERSt6vectorIS1_SaIS1_EE(%"struct.llvm::InlineAsm::ConstraintInfo"* nocapture %this, i64 %Str.0, i64 %Str.1, %"struct.std::vector<llvm::InlineAsm::ConstraintInfo,std::allocator<llvm::InlineAsm::ConstraintInfo> >"* nocapture %ConstraintsSoFar) nounwind ssp align 2 {
entry:
  br i1 undef, label %bb56, label %bb27.outer

bb8:                                              ; preds = %bb27.outer108, %bb13
  switch i8 undef, label %bb27.outer [
    i8 35, label %bb56
    i8 37, label %bb14
    i8 38, label %bb10
    i8 42, label %bb56
  ]

bb27.outer:                                       ; preds = %bb8, %entry
  %I.2.ph = phi i8* [ undef, %entry ], [ %I.2.ph109, %bb8 ] ; <i8*> [#uses=2]
  br label %bb27.outer108

bb10:                                             ; preds = %bb8
  %toBool = icmp eq i8 0, 0                       ; <i1> [#uses=1]
  %or.cond = and i1 undef, %toBool                ; <i1> [#uses=1]
  br i1 %or.cond, label %bb13, label %bb56

bb13:                                             ; preds = %bb10
  br i1 undef, label %bb27.outer108, label %bb8

bb14:                                             ; preds = %bb8
  ret i8 1

bb27.outer108:                                    ; preds = %bb13, %bb27.outer
  %I.2.ph109 = getelementptr i8* %I.2.ph, i64 undef ; <i8*> [#uses=1]
  %scevgep = getelementptr i8* %I.2.ph, i64 undef ; <i8*> [#uses=0]
  br label %bb8

bb56:                                             ; preds = %bb10, %bb8, %bb8, %entry
  ret i8 1
}
