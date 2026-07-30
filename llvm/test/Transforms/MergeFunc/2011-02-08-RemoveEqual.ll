; RUN: opt -passes=mergefunc -disable-output < %s
; This used to crash.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i386-pc-linux-gnu"

%"struct.kc::impl_Ccode_option" = type { %"struct.kc::impl_abstract_phylum" }
%"struct.kc::impl_CexpressionDQ" = type { %"struct.kc::impl_Ccode_option", ptr, ptr }
%"struct.kc::impl_Ctext" = type { %"struct.kc::impl_Ccode_option", i32, ptr, ptr, ptr }
%"struct.kc::impl_Ctext_elem" = type { %"struct.kc::impl_abstract_phylum", i32, ptr }
%"struct.kc::impl_ID" = type { %"struct.kc::impl_abstract_phylum", ptr, ptr, i32, ptr }
%"struct.kc::impl_abstract_phylum" = type { ptr }
%"struct.kc::impl_ac_abstract_declarator_AcAbsdeclDirdecl" = type { %"struct.kc::impl_Ccode_option", ptr, ptr }
%"struct.kc::impl_casestring__Str" = type { %"struct.kc::impl_abstract_phylum", ptr }
%"struct.kc::impl_elem_patternrepresentation" = type { %"struct.kc::impl_abstract_phylum", i32, ptr, ptr }
%"struct.kc::impl_fileline" = type { %"struct.kc::impl_abstract_phylum", ptr, i32 }
%"struct.kc::impl_fileline_FileLine" = type { %"struct.kc::impl_fileline" }
%"struct.kc::impl_outmostpatterns" = type { %"struct.kc::impl_Ccode_option", ptr, ptr }
%"struct.kc::impl_withcaseinfo_Withcaseinfo" = type { %"struct.kc::impl_Ccode_option", ptr, ptr, ptr }

@_ZTVN2kc13impl_filelineE = external constant [13 x ptr], align 32
@.str = external constant [1 x i8], align 1
@_ZTVN2kc22impl_fileline_FileLineE = external constant [13 x ptr], align 32

define void @_ZN2kc22impl_fileline_FileLineC2EPNS_20impl_casestring__StrEi(ptr %this, ptr %_file, i32 %_line) align 2 personality ptr @__gxx_personality_v0 {
entry:
  %this_addr = alloca ptr, align 4
  %_file_addr = alloca ptr, align 4
  %_line_addr = alloca i32, align 4
  %save_filt.150 = alloca i32
  %save_eptr.149 = alloca ptr
  %iftmp.99 = alloca ptr
  %eh_exception = alloca ptr
  %eh_selector = alloca i32
  %"alloca point" = bitcast i32 0 to i32
  store ptr %this, ptr %this_addr
  store ptr %_file, ptr %_file_addr
  store i32 %_line, ptr %_line_addr
  %0 = load ptr, ptr %this_addr, align 4
  call void @_ZN2kc13impl_filelineC2Ev() nounwind
  %1 = load ptr, ptr %this_addr, align 4
  store ptr getelementptr inbounds ([13 x ptr], ptr @_ZTVN2kc22impl_fileline_FileLineE, i32 0, i32 2), ptr %1, align 4
  %2 = load ptr, ptr %_file_addr, align 4
  %3 = icmp eq ptr %2, null
  br i1 %3, label %bb, label %bb1

bb:                                               ; preds = %entry
  %4 = invoke ptr @_ZN2kc12mkcasestringEPKci()
          to label %invcont unwind label %lpad

invcont:                                          ; preds = %bb
  store ptr %4, ptr %iftmp.99, align 4
  br label %bb2

bb1:                                              ; preds = %entry
  %5 = load ptr, ptr %_file_addr, align 4
  store ptr %5, ptr %iftmp.99, align 4
  br label %bb2

bb2:                                              ; preds = %bb1, %invcont
  %6 = load ptr, ptr %this_addr, align 4
  %7 = getelementptr inbounds %"struct.kc::impl_fileline", ptr %6, i32 0, i32 1
  %8 = load ptr, ptr %iftmp.99, align 4
  store ptr %8, ptr %7, align 4
  %9 = load ptr, ptr %this_addr, align 4
  %10 = getelementptr inbounds %"struct.kc::impl_fileline", ptr %9, i32 0, i32 2
  %11 = load i32, ptr %_line_addr, align 4
  store i32 %11, ptr %10, align 4
  ret void

lpad:                                             ; preds = %bb
  %eh_ptr = landingpad { ptr, i32 }
              cleanup
  %exn = extractvalue { ptr, i32 } %eh_ptr, 0
  store ptr %exn, ptr %eh_exception
  %eh_ptr4 = load ptr, ptr %eh_exception
  %eh_select5 = extractvalue { ptr, i32 } %eh_ptr, 1
  store i32 %eh_select5, ptr %eh_selector
  %eh_select = load i32, ptr %eh_selector
  store i32 %eh_select, ptr %save_filt.150, align 4
  %eh_value = load ptr, ptr %eh_exception
  store ptr %eh_value, ptr %save_eptr.149, align 4
  %12 = load ptr, ptr %this_addr, align 4
  call void @_ZN2kc13impl_filelineD2Ev(ptr %12) nounwind
  %13 = load ptr, ptr %save_eptr.149, align 4
  store ptr %13, ptr %eh_exception, align 4
  %14 = load i32, ptr %save_filt.150, align 4
  store i32 %14, ptr %eh_selector, align 4
  %eh_ptr6 = load ptr, ptr %eh_exception
  call void @_Unwind_Resume_or_Rethrow()
  unreachable
}

declare void @_ZN2kc13impl_filelineC2Ev() nounwind align 2

define void @_ZN2kc13impl_filelineD1Ev(ptr %this) nounwind align 2 {
entry:
  %this_addr = alloca ptr, align 4
  %"alloca point" = bitcast i32 0 to i32
  store ptr %this, ptr %this_addr
  %0 = load ptr, ptr %this_addr, align 4
  store ptr getelementptr inbounds ([13 x ptr], ptr @_ZTVN2kc13impl_filelineE, i32 0, i32 2), ptr %0, align 4
  %1 = trunc i32 0 to i8
  %toBool = icmp ne i8 %1, 0
  br i1 %toBool, label %bb1, label %return

bb1:                                              ; preds = %entry
  %2 = load ptr, ptr %this_addr, align 4
  call void @_ZdlPv() nounwind
  br label %return

return:                                           ; preds = %bb1, %entry
  ret void
}

declare void @_ZdlPv() nounwind

define void @_ZN2kc13impl_filelineD2Ev(ptr %this) nounwind align 2 {
entry:
  %this_addr = alloca ptr, align 4
  %"alloca point" = bitcast i32 0 to i32
  store ptr %this, ptr %this_addr
  %0 = load ptr, ptr %this_addr, align 4
  store ptr getelementptr inbounds ([13 x ptr], ptr @_ZTVN2kc13impl_filelineE, i32 0, i32 2), ptr %0, align 4
  %1 = trunc i32 0 to i8
  %toBool = icmp ne i8 %1, 0
  br i1 %toBool, label %bb1, label %return

bb1:                                              ; preds = %entry
  %2 = load ptr, ptr %this_addr, align 4
  call void @_ZdlPv() nounwind
  br label %return

return:                                           ; preds = %bb1, %entry
  ret void
}

define void @_ZN2kc22impl_fileline_FileLineC1EPNS_20impl_casestring__StrEi(ptr %this, ptr %_file, i32 %_line) align 2 personality ptr @__gxx_personality_v0 {
entry:
  %this_addr = alloca ptr, align 4
  %_file_addr = alloca ptr, align 4
  %_line_addr = alloca i32, align 4
  %save_filt.148 = alloca i32
  %save_eptr.147 = alloca ptr
  %iftmp.99 = alloca ptr
  %eh_exception = alloca ptr
  %eh_selector = alloca i32
  %"alloca point" = bitcast i32 0 to i32
  store ptr %this, ptr %this_addr
  store ptr %_file, ptr %_file_addr
  store i32 %_line, ptr %_line_addr
  %0 = load ptr, ptr %this_addr, align 4
  call void @_ZN2kc13impl_filelineC2Ev() nounwind
  %1 = load ptr, ptr %this_addr, align 4
  store ptr getelementptr inbounds ([13 x ptr], ptr @_ZTVN2kc22impl_fileline_FileLineE, i32 0, i32 2), ptr %1, align 4
  %2 = load ptr, ptr %_file_addr, align 4
  %3 = icmp eq ptr %2, null
  br i1 %3, label %bb, label %bb1

bb:                                               ; preds = %entry
  %4 = invoke ptr @_ZN2kc12mkcasestringEPKci()
          to label %invcont unwind label %lpad

invcont:                                          ; preds = %bb
  store ptr %4, ptr %iftmp.99, align 4
  br label %bb2

bb1:                                              ; preds = %entry
  %5 = load ptr, ptr %_file_addr, align 4
  store ptr %5, ptr %iftmp.99, align 4
  br label %bb2

bb2:                                              ; preds = %bb1, %invcont
  %6 = load ptr, ptr %this_addr, align 4
  %7 = getelementptr inbounds %"struct.kc::impl_fileline", ptr %6, i32 0, i32 1
  %8 = load ptr, ptr %iftmp.99, align 4
  store ptr %8, ptr %7, align 4
  %9 = load ptr, ptr %this_addr, align 4
  %10 = getelementptr inbounds %"struct.kc::impl_fileline", ptr %9, i32 0, i32 2
  %11 = load i32, ptr %_line_addr, align 4
  store i32 %11, ptr %10, align 4
  ret void

lpad:                                             ; preds = %bb
  %eh_ptr = landingpad { ptr, i32 }
              cleanup
  %exn = extractvalue { ptr, i32 } %eh_ptr, 0
  store ptr %exn, ptr %eh_exception
  %eh_ptr4 = load ptr, ptr %eh_exception
  %eh_select5 = extractvalue { ptr, i32 } %eh_ptr, 1
  store i32 %eh_select5, ptr %eh_selector
  %eh_select = load i32, ptr %eh_selector
  store i32 %eh_select, ptr %save_filt.148, align 4
  %eh_value = load ptr, ptr %eh_exception
  store ptr %eh_value, ptr %save_eptr.147, align 4
  %12 = load ptr, ptr %this_addr, align 4
  call void @_ZN2kc13impl_filelineD2Ev(ptr %12) nounwind
  %13 = load ptr, ptr %save_eptr.147, align 4
  store ptr %13, ptr %eh_exception, align 4
  %14 = load i32, ptr %save_filt.148, align 4
  store i32 %14, ptr %eh_selector, align 4
  %eh_ptr6 = load ptr, ptr %eh_exception
  call void @_Unwind_Resume_or_Rethrow()
  unreachable
}

declare i32 @__gxx_personality_v0(...)

declare void @_Unwind_Resume_or_Rethrow()

define void @_ZN2kc21printer_functor_classC2Ev(ptr %this) nounwind align 2 {
entry:
  unreachable
}

define ptr @_ZN2kc11phylum_castIPNS_17impl_withcaseinfoES1_EET_PT0_(ptr %t) nounwind {
entry:
  ret ptr null
}

define ptr @_ZNK2kc43impl_ac_direct_declarator_AcDirectDeclProto9subphylumEi(ptr %this, i32 %no) nounwind align 2 {
entry:
  ret ptr undef
}

define void @_ZN2kc30impl_withcaseinfo_WithcaseinfoD0Ev(ptr %this) nounwind align 2 {
entry:
  unreachable
}

define void @_ZN2kc30impl_withcaseinfo_WithcaseinfoC1EPNS_26impl_patternrepresentationES2_PNS_10impl_CtextE(ptr %this, ptr %_patternrepresentation_1, ptr %_patternrepresentation_2, ptr %_Ctext_1) nounwind align 2 {
entry:
  unreachable
}

define void @_ZN2kc21impl_rewriteviewsinfoC2EPNS_20impl_rewriteviewinfoEPS0_(ptr %this, ptr %p1, ptr %p2) nounwind align 2 {
entry:
  unreachable
}

define ptr @_ZN2kc11phylum_castIPNS_9impl_termENS_20impl_abstract_phylumEEET_PT0_(ptr %t) nounwind {
entry:
  unreachable
}

define void @_ZN2kc27impl_ac_parameter_type_listD2Ev(ptr %this) nounwind align 2 {
entry:
  ret void
}

define void @_ZN2kc21impl_ac_operator_nameD2Ev(ptr %this) nounwind align 2 {
entry:
  ret void
}

declare ptr @_ZN2kc12mkcasestringEPKci()
