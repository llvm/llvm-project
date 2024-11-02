; RUN: opt -passes=mergefunc -disable-output < %s
; This used to crash.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i386-pc-linux-gnu"

%"struct.kc::impl_Ccode_option" = type { %"struct.kc::impl_abstract_phylum" }
%"struct.kc::impl_CexpressionDQ" = type { %"struct.kc::impl_Ccode_option", %"struct.kc::impl_Ccode_option"*, %"struct.kc::impl_CexpressionDQ"* }
%"struct.kc::impl_Ctext" = type { %"struct.kc::impl_Ccode_option", i32, %"struct.kc::impl_casestring__Str"*, %"struct.kc::impl_Ctext_elem"*, %"struct.kc::impl_Ctext"* }
%"struct.kc::impl_Ctext_elem" = type { %"struct.kc::impl_abstract_phylum", i32, %"struct.kc::impl_casestring__Str"* }
%"struct.kc::impl_ID" = type { %"struct.kc::impl_abstract_phylum", %"struct.kc::impl_Ccode_option"*, %"struct.kc::impl_casestring__Str"*, i32, %"struct.kc::impl_casestring__Str"* }
%"struct.kc::impl_abstract_phylum" = type { i32 (...)** }
%"struct.kc::impl_ac_abstract_declarator_AcAbsdeclDirdecl" = type { %"struct.kc::impl_Ccode_option", %"struct.kc::impl_Ccode_option"*, %"struct.kc::impl_Ccode_option"* }
%"struct.kc::impl_casestring__Str" = type { %"struct.kc::impl_abstract_phylum", i8* }
%"struct.kc::impl_elem_patternrepresentation" = type { %"struct.kc::impl_abstract_phylum", i32, %"struct.kc::impl_casestring__Str"*, %"struct.kc::impl_ID"* }
%"struct.kc::impl_fileline" = type { %"struct.kc::impl_abstract_phylum", %"struct.kc::impl_casestring__Str"*, i32 }
%"struct.kc::impl_fileline_FileLine" = type { %"struct.kc::impl_fileline" }
%"struct.kc::impl_outmostpatterns" = type { %"struct.kc::impl_Ccode_option", %"struct.kc::impl_elem_patternrepresentation"*, %"struct.kc::impl_outmostpatterns"* }
%"struct.kc::impl_withcaseinfo_Withcaseinfo" = type { %"struct.kc::impl_Ccode_option", %"struct.kc::impl_outmostpatterns"*, %"struct.kc::impl_outmostpatterns"*, %"struct.kc::impl_Ctext"* }

@_ZTVN2kc13impl_filelineE = external constant [13 x i32 (...)*], align 32
@.str = external constant [1 x i8], align 1
@_ZTVN2kc22impl_fileline_FileLineE = external constant [13 x i32 (...)*], align 32

define void @_ZN2kc22impl_fileline_FileLineC2EPNS_20impl_casestring__StrEi(%"struct.kc::impl_fileline_FileLine"* %this, %"struct.kc::impl_casestring__Str"* %_file, i32 %_line) align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %this_addr = alloca %"struct.kc::impl_fileline_FileLine"*, align 4
  %_file_addr = alloca %"struct.kc::impl_casestring__Str"*, align 4
  %_line_addr = alloca i32, align 4
  %save_filt.150 = alloca i32
  %save_eptr.149 = alloca i8*
  %iftmp.99 = alloca %"struct.kc::impl_casestring__Str"*
  %eh_exception = alloca i8*
  %eh_selector = alloca i32
  %"alloca point" = bitcast i32 0 to i32
  store %"struct.kc::impl_fileline_FileLine"* %this, %"struct.kc::impl_fileline_FileLine"** %this_addr
  store %"struct.kc::impl_casestring__Str"* %_file, %"struct.kc::impl_casestring__Str"** %_file_addr
  store i32 %_line, i32* %_line_addr
  %0 = load %"struct.kc::impl_fileline_FileLine"*, %"struct.kc::impl_fileline_FileLine"** %this_addr, align 4
  %1 = getelementptr inbounds %"struct.kc::impl_fileline_FileLine", %"struct.kc::impl_fileline_FileLine"* %0, i32 0, i32 0
  call void @_ZN2kc13impl_filelineC2Ev() nounwind
  %2 = load %"struct.kc::impl_fileline_FileLine"*, %"struct.kc::impl_fileline_FileLine"** %this_addr, align 4
  %3 = getelementptr inbounds %"struct.kc::impl_fileline_FileLine", %"struct.kc::impl_fileline_FileLine"* %2, i32 0, i32 0
  %4 = getelementptr inbounds %"struct.kc::impl_fileline", %"struct.kc::impl_fileline"* %3, i32 0, i32 0
  %5 = getelementptr inbounds %"struct.kc::impl_abstract_phylum", %"struct.kc::impl_abstract_phylum"* %4, i32 0, i32 0
  store i32 (...)** getelementptr inbounds ([13 x i32 (...)*], [13 x i32 (...)*]* @_ZTVN2kc22impl_fileline_FileLineE, i32 0, i32 2), i32 (...)*** %5, align 4
  %6 = load %"struct.kc::impl_casestring__Str"*, %"struct.kc::impl_casestring__Str"** %_file_addr, align 4
  %7 = icmp eq %"struct.kc::impl_casestring__Str"* %6, null
  br i1 %7, label %bb, label %bb1

bb:                                               ; preds = %entry
  %8 = invoke %"struct.kc::impl_casestring__Str"* @_ZN2kc12mkcasestringEPKci()
          to label %invcont unwind label %lpad

invcont:                                          ; preds = %bb
  store %"struct.kc::impl_casestring__Str"* %8, %"struct.kc::impl_casestring__Str"** %iftmp.99, align 4
  br label %bb2

bb1:                                              ; preds = %entry
  %9 = load %"struct.kc::impl_casestring__Str"*, %"struct.kc::impl_casestring__Str"** %_file_addr, align 4
  store %"struct.kc::impl_casestring__Str"* %9, %"struct.kc::impl_casestring__Str"** %iftmp.99, align 4
  br label %bb2

bb2:                                              ; preds = %bb1, %invcont
  %10 = load %"struct.kc::impl_fileline_FileLine"*, %"struct.kc::impl_fileline_FileLine"** %this_addr, align 4
  %11 = getelementptr inbounds %"struct.kc::impl_fileline_FileLine", %"struct.kc::impl_fileline_FileLine"* %10, i32 0, i32 0
  %12 = getelementptr inbounds %"struct.kc::impl_fileline", %"struct.kc::impl_fileline"* %11, i32 0, i32 1
  %13 = load %"struct.kc::impl_casestring__Str"*, %"struct.kc::impl_casestring__Str"** %iftmp.99, align 4
  store %"struct.kc::impl_casestring__Str"* %13, %"struct.kc::impl_casestring__Str"** %12, align 4
  %14 = load %"struct.kc::impl_fileline_FileLine"*, %"struct.kc::impl_fileline_FileLine"** %this_addr, align 4
  %15 = getelementptr inbounds %"struct.kc::impl_fileline_FileLine", %"struct.kc::impl_fileline_FileLine"* %14, i32 0, i32 0
  %16 = getelementptr inbounds %"struct.kc::impl_fileline", %"struct.kc::impl_fileline"* %15, i32 0, i32 2
  %17 = load i32, i32* %_line_addr, align 4
  store i32 %17, i32* %16, align 4
  ret void

lpad:                                             ; preds = %bb
  %eh_ptr = landingpad { i8*, i32 }
              cleanup
  %exn = extractvalue { i8*, i32 } %eh_ptr, 0
  store i8* %exn, i8** %eh_exception
  %eh_ptr4 = load i8*, i8** %eh_exception
  %eh_select5 = extractvalue { i8*, i32 } %eh_ptr, 1
  store i32 %eh_select5, i32* %eh_selector
  %eh_select = load i32, i32* %eh_selector
  store i32 %eh_select, i32* %save_filt.150, align 4
  %eh_value = load i8*, i8** %eh_exception
  store i8* %eh_value, i8** %save_eptr.149, align 4
  %18 = load %"struct.kc::impl_fileline_FileLine"*, %"struct.kc::impl_fileline_FileLine"** %this_addr, align 4
  %19 = bitcast %"struct.kc::impl_fileline_FileLine"* %18 to %"struct.kc::impl_fileline"*
  call void @_ZN2kc13impl_filelineD2Ev(%"struct.kc::impl_fileline"* %19) nounwind
  %20 = load i8*, i8** %save_eptr.149, align 4
  store i8* %20, i8** %eh_exception, align 4
  %21 = load i32, i32* %save_filt.150, align 4
  store i32 %21, i32* %eh_selector, align 4
  %eh_ptr6 = load i8*, i8** %eh_exception
  call void @_Unwind_Resume_or_Rethrow()
  unreachable
}

declare void @_ZN2kc13impl_filelineC2Ev() nounwind align 2

define void @_ZN2kc13impl_filelineD1Ev(%"struct.kc::impl_fileline"* %this) nounwind align 2 {
entry:
  %this_addr = alloca %"struct.kc::impl_fileline"*, align 4
  %"alloca point" = bitcast i32 0 to i32
  store %"struct.kc::impl_fileline"* %this, %"struct.kc::impl_fileline"** %this_addr
  %0 = load %"struct.kc::impl_fileline"*, %"struct.kc::impl_fileline"** %this_addr, align 4
  %1 = getelementptr inbounds %"struct.kc::impl_fileline", %"struct.kc::impl_fileline"* %0, i32 0, i32 0
  %2 = getelementptr inbounds %"struct.kc::impl_abstract_phylum", %"struct.kc::impl_abstract_phylum"* %1, i32 0, i32 0
  store i32 (...)** getelementptr inbounds ([13 x i32 (...)*], [13 x i32 (...)*]* @_ZTVN2kc13impl_filelineE, i32 0, i32 2), i32 (...)*** %2, align 4
  %3 = trunc i32 0 to i8
  %toBool = icmp ne i8 %3, 0
  br i1 %toBool, label %bb1, label %return

bb1:                                              ; preds = %entry
  %4 = load %"struct.kc::impl_fileline"*, %"struct.kc::impl_fileline"** %this_addr, align 4
  %5 = bitcast %"struct.kc::impl_fileline"* %4 to i8*
  call void @_ZdlPv() nounwind
  br label %return

return:                                           ; preds = %bb1, %entry
  ret void
}

declare void @_ZdlPv() nounwind

define void @_ZN2kc13impl_filelineD2Ev(%"struct.kc::impl_fileline"* %this) nounwind align 2 {
entry:
  %this_addr = alloca %"struct.kc::impl_fileline"*, align 4
  %"alloca point" = bitcast i32 0 to i32
  store %"struct.kc::impl_fileline"* %this, %"struct.kc::impl_fileline"** %this_addr
  %0 = load %"struct.kc::impl_fileline"*, %"struct.kc::impl_fileline"** %this_addr, align 4
  %1 = getelementptr inbounds %"struct.kc::impl_fileline", %"struct.kc::impl_fileline"* %0, i32 0, i32 0
  %2 = getelementptr inbounds %"struct.kc::impl_abstract_phylum", %"struct.kc::impl_abstract_phylum"* %1, i32 0, i32 0
  store i32 (...)** getelementptr inbounds ([13 x i32 (...)*], [13 x i32 (...)*]* @_ZTVN2kc13impl_filelineE, i32 0, i32 2), i32 (...)*** %2, align 4
  %3 = trunc i32 0 to i8
  %toBool = icmp ne i8 %3, 0
  br i1 %toBool, label %bb1, label %return

bb1:                                              ; preds = %entry
  %4 = load %"struct.kc::impl_fileline"*, %"struct.kc::impl_fileline"** %this_addr, align 4
  %5 = bitcast %"struct.kc::impl_fileline"* %4 to i8*
  call void @_ZdlPv() nounwind
  br label %return

return:                                           ; preds = %bb1, %entry
  ret void
}

define void @_ZN2kc22impl_fileline_FileLineC1EPNS_20impl_casestring__StrEi(%"struct.kc::impl_fileline_FileLine"* %this, %"struct.kc::impl_casestring__Str"* %_file, i32 %_line) align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %this_addr = alloca %"struct.kc::impl_fileline_FileLine"*, align 4
  %_file_addr = alloca %"struct.kc::impl_casestring__Str"*, align 4
  %_line_addr = alloca i32, align 4
  %save_filt.148 = alloca i32
  %save_eptr.147 = alloca i8*
  %iftmp.99 = alloca %"struct.kc::impl_casestring__Str"*
  %eh_exception = alloca i8*
  %eh_selector = alloca i32
  %"alloca point" = bitcast i32 0 to i32
  store %"struct.kc::impl_fileline_FileLine"* %this, %"struct.kc::impl_fileline_FileLine"** %this_addr
  store %"struct.kc::impl_casestring__Str"* %_file, %"struct.kc::impl_casestring__Str"** %_file_addr
  store i32 %_line, i32* %_line_addr
  %0 = load %"struct.kc::impl_fileline_FileLine"*, %"struct.kc::impl_fileline_FileLine"** %this_addr, align 4
  %1 = getelementptr inbounds %"struct.kc::impl_fileline_FileLine", %"struct.kc::impl_fileline_FileLine"* %0, i32 0, i32 0
  call void @_ZN2kc13impl_filelineC2Ev() nounwind
  %2 = load %"struct.kc::impl_fileline_FileLine"*, %"struct.kc::impl_fileline_FileLine"** %this_addr, align 4
  %3 = getelementptr inbounds %"struct.kc::impl_fileline_FileLine", %"struct.kc::impl_fileline_FileLine"* %2, i32 0, i32 0
  %4 = getelementptr inbounds %"struct.kc::impl_fileline", %"struct.kc::impl_fileline"* %3, i32 0, i32 0
  %5 = getelementptr inbounds %"struct.kc::impl_abstract_phylum", %"struct.kc::impl_abstract_phylum"* %4, i32 0, i32 0
  store i32 (...)** getelementptr inbounds ([13 x i32 (...)*], [13 x i32 (...)*]* @_ZTVN2kc22impl_fileline_FileLineE, i32 0, i32 2), i32 (...)*** %5, align 4
  %6 = load %"struct.kc::impl_casestring__Str"*, %"struct.kc::impl_casestring__Str"** %_file_addr, align 4
  %7 = icmp eq %"struct.kc::impl_casestring__Str"* %6, null
  br i1 %7, label %bb, label %bb1

bb:                                               ; preds = %entry
  %8 = invoke %"struct.kc::impl_casestring__Str"* @_ZN2kc12mkcasestringEPKci()
          to label %invcont unwind label %lpad

invcont:                                          ; preds = %bb
  store %"struct.kc::impl_casestring__Str"* %8, %"struct.kc::impl_casestring__Str"** %iftmp.99, align 4
  br label %bb2

bb1:                                              ; preds = %entry
  %9 = load %"struct.kc::impl_casestring__Str"*, %"struct.kc::impl_casestring__Str"** %_file_addr, align 4
  store %"struct.kc::impl_casestring__Str"* %9, %"struct.kc::impl_casestring__Str"** %iftmp.99, align 4
  br label %bb2

bb2:                                              ; preds = %bb1, %invcont
  %10 = load %"struct.kc::impl_fileline_FileLine"*, %"struct.kc::impl_fileline_FileLine"** %this_addr, align 4
  %11 = getelementptr inbounds %"struct.kc::impl_fileline_FileLine", %"struct.kc::impl_fileline_FileLine"* %10, i32 0, i32 0
  %12 = getelementptr inbounds %"struct.kc::impl_fileline", %"struct.kc::impl_fileline"* %11, i32 0, i32 1
  %13 = load %"struct.kc::impl_casestring__Str"*, %"struct.kc::impl_casestring__Str"** %iftmp.99, align 4
  store %"struct.kc::impl_casestring__Str"* %13, %"struct.kc::impl_casestring__Str"** %12, align 4
  %14 = load %"struct.kc::impl_fileline_FileLine"*, %"struct.kc::impl_fileline_FileLine"** %this_addr, align 4
  %15 = getelementptr inbounds %"struct.kc::impl_fileline_FileLine", %"struct.kc::impl_fileline_FileLine"* %14, i32 0, i32 0
  %16 = getelementptr inbounds %"struct.kc::impl_fileline", %"struct.kc::impl_fileline"* %15, i32 0, i32 2
  %17 = load i32, i32* %_line_addr, align 4
  store i32 %17, i32* %16, align 4
  ret void

lpad:                                             ; preds = %bb
  %eh_ptr = landingpad { i8*, i32 }
              cleanup
  %exn = extractvalue { i8*, i32 } %eh_ptr, 0
  store i8* %exn, i8** %eh_exception
  %eh_ptr4 = load i8*, i8** %eh_exception
  %eh_select5 = extractvalue { i8*, i32 } %eh_ptr, 1
  store i32 %eh_select5, i32* %eh_selector
  %eh_select = load i32, i32* %eh_selector
  store i32 %eh_select, i32* %save_filt.148, align 4
  %eh_value = load i8*, i8** %eh_exception
  store i8* %eh_value, i8** %save_eptr.147, align 4
  %18 = load %"struct.kc::impl_fileline_FileLine"*, %"struct.kc::impl_fileline_FileLine"** %this_addr, align 4
  %19 = bitcast %"struct.kc::impl_fileline_FileLine"* %18 to %"struct.kc::impl_fileline"*
  call void @_ZN2kc13impl_filelineD2Ev(%"struct.kc::impl_fileline"* %19) nounwind
  %20 = load i8*, i8** %save_eptr.147, align 4
  store i8* %20, i8** %eh_exception, align 4
  %21 = load i32, i32* %save_filt.148, align 4
  store i32 %21, i32* %eh_selector, align 4
  %eh_ptr6 = load i8*, i8** %eh_exception
  call void @_Unwind_Resume_or_Rethrow()
  unreachable
}

declare i32 @__gxx_personality_v0(...)

declare void @_Unwind_Resume_or_Rethrow()

define void @_ZN2kc21printer_functor_classC2Ev(%"struct.kc::impl_abstract_phylum"* %this) nounwind align 2 {
entry:
  unreachable
}

define %"struct.kc::impl_Ccode_option"* @_ZN2kc11phylum_castIPNS_17impl_withcaseinfoES1_EET_PT0_(%"struct.kc::impl_Ccode_option"* %t) nounwind {
entry:
  ret %"struct.kc::impl_Ccode_option"* null
}

define %"struct.kc::impl_abstract_phylum"* @_ZNK2kc43impl_ac_direct_declarator_AcDirectDeclProto9subphylumEi(%"struct.kc::impl_ac_abstract_declarator_AcAbsdeclDirdecl"* %this, i32 %no) nounwind align 2 {
entry:
  ret %"struct.kc::impl_abstract_phylum"* undef
}

define void @_ZN2kc30impl_withcaseinfo_WithcaseinfoD0Ev(%"struct.kc::impl_withcaseinfo_Withcaseinfo"* %this) nounwind align 2 {
entry:
  unreachable
}

define void @_ZN2kc30impl_withcaseinfo_WithcaseinfoC1EPNS_26impl_patternrepresentationES2_PNS_10impl_CtextE(%"struct.kc::impl_withcaseinfo_Withcaseinfo"* %this, %"struct.kc::impl_outmostpatterns"* %_patternrepresentation_1, %"struct.kc::impl_outmostpatterns"* %_patternrepresentation_2, %"struct.kc::impl_Ctext"* %_Ctext_1) nounwind align 2 {
entry:
  unreachable
}

define void @_ZN2kc21impl_rewriteviewsinfoC2EPNS_20impl_rewriteviewinfoEPS0_(%"struct.kc::impl_CexpressionDQ"* %this, %"struct.kc::impl_Ccode_option"* %p1, %"struct.kc::impl_CexpressionDQ"* %p2) nounwind align 2 {
entry:
  unreachable
}

define %"struct.kc::impl_Ctext_elem"* @_ZN2kc11phylum_castIPNS_9impl_termENS_20impl_abstract_phylumEEET_PT0_(%"struct.kc::impl_abstract_phylum"* %t) nounwind {
entry:
  unreachable
}

define void @_ZN2kc27impl_ac_parameter_type_listD2Ev(%"struct.kc::impl_Ccode_option"* %this) nounwind align 2 {
entry:
  ret void
}

define void @_ZN2kc21impl_ac_operator_nameD2Ev(%"struct.kc::impl_Ctext_elem"* %this) nounwind align 2 {
entry:
  ret void
}

declare %"struct.kc::impl_casestring__Str"* @_ZN2kc12mkcasestringEPKci()
