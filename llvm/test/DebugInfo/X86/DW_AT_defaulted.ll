; RUN: llc < %s -filetype=obj -o %t
; RUN: llvm-dwarfdump -v %t | FileCheck %s

; C++ source to regenerate:
; class DefaultedInline {
; public:
;   DefaultedInline() = default;
;   ~DefaultedInline() = default;
;
;   DefaultedInline(const DefaultedInline &) = default;
;   DefaultedInline &operator=(const DefaultedInline &) = default;
;
;   DefaultedInline(DefaultedInline &&) = default;
;   DefaultedInline &operator=(DefaultedInline &&) = default;
; };
;
; class DefaultedOutline {
; public:
;   DefaultedOutline();
;   ~DefaultedOutline();
;
;   DefaultedOutline(const DefaultedOutline &);
;   DefaultedOutline &operator=(const DefaultedOutline &);
;
;   DefaultedOutline(DefaultedOutline &&);
;   DefaultedOutline &operator=(DefaultedOutline &&);
; };
;
; DefaultedOutline::DefaultedOutline() = default;
; DefaultedOutline::~DefaultedOutline() = default;
;
; DefaultedOutline::DefaultedOutline(const DefaultedOutline &) = default;
; DefaultedOutline &
; DefaultedOutline::operator=(const DefaultedOutline &) = default;
;
; DefaultedOutline::DefaultedOutline(DefaultedOutline &&) = default;
; DefaultedOutline &DefaultedOutline::operator=(DefaultedOutline &&) = default;
;
; class NeverDefaulted {
; public:
;   NeverDefaulted() {}
;   ~NeverDefaulted() {}
;
;   NeverDefaulted(const NeverDefaulted &) {}
;   NeverDefaulted &operator=(const NeverDefaulted &) { return *this; }
;
;   NeverDefaulted(NeverDefaulted &&) {}
;   NeverDefaulted &operator=(NeverDefaulted &&) { return *this; }
; };
;
; template <int N> class DefaultedInlineWithTemplate {
; public:
;   char m[N];
;
;   DefaultedInlineWithTemplate() = default;
;   ~DefaultedInlineWithTemplate() = default;
;
;   DefaultedInlineWithTemplate(const DefaultedInlineWithTemplate &) = default;
;   DefaultedInlineWithTemplate &
;   operator=(const DefaultedInlineWithTemplate &) = default;
;
;   DefaultedInlineWithTemplate(DefaultedInlineWithTemplate &&) = default;
;   DefaultedInlineWithTemplate &
;   operator=(DefaultedInlineWithTemplate &&) = default;
; };
;
; int main() {
;   DefaultedInline a;
;   DefaultedOutline b;
;   NeverDefaulted c;
;   DefaultedInlineWithTemplate<6> d;
;   DefaultedInlineWithTemplate<7> e;
;   return 0;
; }
; $ clang++ -O0 -g -gdwarf-5 debug-info-defaulted.cpp -S -emit-llvm

; CHECK: .debug_abbrev contents:

; CHECK: [11] DW_TAG_subprogram DW_CHILDREN_yes
; CHECK: DW_AT_defaulted DW_FORM_data1

; CHECK: [13] DW_TAG_subprogram DW_CHILDREN_yes
; CHECK: DW_AT_defaulted DW_FORM_data1

; CHECK: [14] DW_TAG_subprogram DW_CHILDREN_yes
; CHECK: DW_AT_defaulted DW_FORM_data1

; CHECK: [15] DW_TAG_subprogram DW_CHILDREN_yes
; CHECK: DW_AT_defaulted DW_FORM_data1

; CHECK: [17] DW_TAG_subprogram DW_CHILDREN_yes
; CHECK: DW_AT_defaulted DW_FORM_data1

; CHECK: [20] DW_TAG_subprogram DW_CHILDREN_yes
; CHECK: DW_AT_defaulted DW_FORM_data1

; CHECK: [21] DW_TAG_subprogram DW_CHILDREN_yes
; CHECK: DW_AT_defaulted DW_FORM_data1

; CHECK: [22] DW_TAG_subprogram DW_CHILDREN_yes
; CHECK: DW_AT_defaulted DW_FORM_data1

; CHECK: [23] DW_TAG_subprogram DW_CHILDREN_yes
; CHECK: DW_AT_defaulted DW_FORM_data1

; CHECK: [24] DW_TAG_subprogram DW_CHILDREN_yes
; CHECK: DW_AT_defaulted DW_FORM_data1

; CHECK: [25] DW_TAG_subprogram DW_CHILDREN_yes
; CHECK: DW_AT_defaulted DW_FORM_data1

; CHECK: .debug_info contents:

; CHECK: DW_TAG_subprogram [11]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN16DefaultedOutlineC2Ev")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_out_of_class)

; CHECK: DW_TAG_subprogram [13]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN16DefaultedOutlineC1Ev")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_out_of_class)

; CHECK: DW_TAG_subprogram [14]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN16DefaultedOutlineD2Ev")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_out_of_class)

; CHECK: DW_TAG_subprogram [15]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN16DefaultedOutlineD1Ev")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_out_of_class)

; CHECK: DW_TAG_subprogram [11]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN16DefaultedOutlineC2ERKS_")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_out_of_class)

; CHECK: DW_TAG_subprogram [13]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN16DefaultedOutlineC1ERKS_")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_out_of_class)

; CHECK: DW_TAG_subprogram [17]
; CHECK-DAG: DW_AT_specification {{.*}} "_ZN16DefaultedOutlineaSERKS_")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_out_of_class)

; CHECK: DW_TAG_subprogram [11]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN16DefaultedOutlineC2EOS_")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_out_of_class)

; CHECK: DW_TAG_subprogram [13]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN16DefaultedOutlineC1EOS_")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_out_of_class)

; CHECK: DW_TAG_subprogram [17]
; CHECK-DAG: DW_AT_specification {{.*}} "_ZN16DefaultedOutlineaSEOS_")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_out_of_class)

; CHECK: DW_TAG_subprogram [20]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN14NeverDefaultedC4Ev")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_no)

; CHECK: DW_TAG_subprogram [20]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN14NeverDefaultedD4Ev")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_no)

; CHECK: DW_TAG_subprogram [20]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN14NeverDefaultedC4ERKS_")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_no)

; CHECK: DW_TAG_subprogram [21]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN14NeverDefaultedaSERKS_")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_no)

; CHECK: DW_TAG_subprogram [20]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN14NeverDefaultedC4EOS_")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_no)

; CHECK: DW_TAG_subprogram [21]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN14NeverDefaultedaSEOS_")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_no)

; CHECK: DW_TAG_subprogram [22]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN14NeverDefaultedC1Ev")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_no)

; CHECK: DW_TAG_subprogram [23]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN14NeverDefaultedD1Ev")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_no)

; CHECK: DW_TAG_subprogram [24]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN14NeverDefaultedC2Ev")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_no)

; CHECK: DW_TAG_subprogram [25]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN14NeverDefaultedD2Ev")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_no)

; CHECK: DW_TAG_subprogram [20]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN15DefaultedInlineC4Ev")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_in_class)

; CHECK: DW_TAG_subprogram [20]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN15DefaultedInlineD4Ev")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_in_class)

; CHECK: DW_TAG_subprogram [20]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN15DefaultedInlineC4ERKS_")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_in_class)

; CHECK: DW_TAG_subprogram [21]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN15DefaultedInlineaSERKS_")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_in_class)

; CHECK: DW_TAG_subprogram [20]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN15DefaultedInlineC4EOS_")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_in_class)

; CHECK: DW_TAG_subprogram [21]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN15DefaultedInlineaSEOS_")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_in_class)

; CHECK: DW_TAG_subprogram [20]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN27DefaultedInlineWithTemplateILi6EEC4Ev")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_in_class)

; CHECK: DW_TAG_subprogram [20]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN27DefaultedInlineWithTemplateILi6EED4Ev")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_in_class)

; CHECK: DW_TAG_subprogram [20]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN27DefaultedInlineWithTemplateILi6EEC4ERKS0_")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_in_class)

; CHECK: DW_TAG_subprogram [21]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN27DefaultedInlineWithTemplateILi6EEaSERKS0_")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_in_class)

; CHECK: DW_TAG_subprogram [20]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN27DefaultedInlineWithTemplateILi6EEC4EOS0_")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_in_class)

; CHECK: DW_TAG_subprogram [21]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN27DefaultedInlineWithTemplateILi6EEaSEOS0_")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_in_class)

; CHECK: DW_TAG_subprogram [20]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN27DefaultedInlineWithTemplateILi7EEC4Ev")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_in_class)

; CHECK: DW_TAG_subprogram [20]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN27DefaultedInlineWithTemplateILi7EED4Ev")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_in_class)

; CHECK: DW_TAG_subprogram [20]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN27DefaultedInlineWithTemplateILi7EEC4ERKS0_")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_in_class)

; CHECK: DW_TAG_subprogram [21]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN27DefaultedInlineWithTemplateILi7EEaSERKS0_")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_in_class)

; CHECK: DW_TAG_subprogram [20]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN27DefaultedInlineWithTemplateILi7EEC4EOS0_")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_in_class)

; CHECK: DW_TAG_subprogram [21]
; CHECK-DAG: DW_AT_linkage_name {{.*}} "_ZN27DefaultedInlineWithTemplateILi7EEaSEOS0_")
; CHECK-DAG: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_in_class)

; ModuleID = 'debug-info-defaulted.cpp'
source_filename = "debug-info-defaulted.cpp"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx26.0.0"

%class.DefaultedInline = type { i8 }
%class.DefaultedOutline = type { i8 }
%class.NeverDefaulted = type { i8 }
%class.DefaultedInlineWithTemplate = type { [6 x i8] }
%class.DefaultedInlineWithTemplate.0 = type { [7 x i8] }

; Function Attrs: mustprogress noinline nounwind optnone ssp uwtable(sync)
define noundef ptr @_ZN16DefaultedOutlineC2Ev(ptr noundef nonnull returned align 1 dereferenceable(1) %this) unnamed_addr #0 !dbg !8 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !33, !DIExpression(), !35)
  %this1 = load ptr, ptr %this.addr, align 8
  ret ptr %this1, !dbg !36
}

; Function Attrs: mustprogress noinline nounwind optnone ssp uwtable(sync)
define noundef ptr @_ZN16DefaultedOutlineC1Ev(ptr noundef nonnull returned align 1 dereferenceable(1) %this) unnamed_addr #0 !dbg !37 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !38, !DIExpression(), !39)
  %this1 = load ptr, ptr %this.addr, align 8
  %call = call noundef ptr @_ZN16DefaultedOutlineC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this1), !dbg !40
  ret ptr %this1, !dbg !40
}

; Function Attrs: mustprogress noinline nounwind optnone ssp uwtable(sync)
define noundef ptr @_ZN16DefaultedOutlineD2Ev(ptr noundef nonnull returned align 1 dereferenceable(1) %this) unnamed_addr #0 !dbg !41 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !44, !DIExpression(), !45)
  %this1 = load ptr, ptr %this.addr, align 8
  ret ptr %this1, !dbg !46
}

; Function Attrs: mustprogress noinline nounwind optnone ssp uwtable(sync)
define noundef ptr @_ZN16DefaultedOutlineD1Ev(ptr noundef nonnull returned align 1 dereferenceable(1) %this) unnamed_addr #0 !dbg !47 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !48, !DIExpression(), !49)
  %this1 = load ptr, ptr %this.addr, align 8
  %call = call noundef ptr @_ZN16DefaultedOutlineD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this1) #3, !dbg !50
  ret ptr %this1, !dbg !50
}

; Function Attrs: mustprogress noinline nounwind optnone ssp uwtable(sync)
define noundef ptr @_ZN16DefaultedOutlineC2ERKS_(ptr noundef nonnull returned align 1 dereferenceable(1) %this, ptr noundef nonnull align 1 dereferenceable(1) %0) unnamed_addr #0 !dbg !51 {
entry:
  %this.addr = alloca ptr, align 8
  %.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !52, !DIExpression(), !53)
  store ptr %0, ptr %.addr, align 8
    #dbg_declare(ptr %.addr, !54, !DIExpression(), !55)
  %this1 = load ptr, ptr %this.addr, align 8
  ret ptr %this1, !dbg !56
}

; Function Attrs: mustprogress noinline nounwind optnone ssp uwtable(sync)
define noundef ptr @_ZN16DefaultedOutlineC1ERKS_(ptr noundef nonnull returned align 1 dereferenceable(1) %this, ptr noundef nonnull align 1 dereferenceable(1) %0) unnamed_addr #0 !dbg !57 {
entry:
  %this.addr = alloca ptr, align 8
  %.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !58, !DIExpression(), !59)
  store ptr %0, ptr %.addr, align 8
    #dbg_declare(ptr %.addr, !60, !DIExpression(), !61)
  %this1 = load ptr, ptr %this.addr, align 8
  %1 = load ptr, ptr %.addr, align 8, !dbg !62
  %call = call noundef ptr @_ZN16DefaultedOutlineC2ERKS_(ptr noundef nonnull align 1 dereferenceable(1) %this1, ptr noundef nonnull align 1 dereferenceable(1) %1), !dbg !62
  ret ptr %this1, !dbg !62
}

; Function Attrs: mustprogress noinline nounwind optnone ssp uwtable(sync)
define noundef nonnull align 1 dereferenceable(1) ptr @_ZN16DefaultedOutlineaSERKS_(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef nonnull align 1 dereferenceable(1) %0) #0 !dbg !63 {
entry:
  %this.addr = alloca ptr, align 8
  %.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !64, !DIExpression(), !65)
  store ptr %0, ptr %.addr, align 8
    #dbg_declare(ptr %.addr, !66, !DIExpression(), !67)
  %this1 = load ptr, ptr %this.addr, align 8
  ret ptr %this1, !dbg !68
}

; Function Attrs: mustprogress noinline nounwind optnone ssp uwtable(sync)
define noundef ptr @_ZN16DefaultedOutlineC2EOS_(ptr noundef nonnull returned align 1 dereferenceable(1) %this, ptr noundef nonnull align 1 dereferenceable(1) %0) unnamed_addr #0 !dbg !70 {
entry:
  %this.addr = alloca ptr, align 8
  %.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !71, !DIExpression(), !72)
  store ptr %0, ptr %.addr, align 8
    #dbg_declare(ptr %.addr, !73, !DIExpression(), !74)
  %this1 = load ptr, ptr %this.addr, align 8
  ret ptr %this1, !dbg !75
}

; Function Attrs: mustprogress noinline nounwind optnone ssp uwtable(sync)
define noundef ptr @_ZN16DefaultedOutlineC1EOS_(ptr noundef nonnull returned align 1 dereferenceable(1) %this, ptr noundef nonnull align 1 dereferenceable(1) %0) unnamed_addr #0 !dbg !76 {
entry:
  %this.addr = alloca ptr, align 8
  %.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !77, !DIExpression(), !78)
  store ptr %0, ptr %.addr, align 8
    #dbg_declare(ptr %.addr, !79, !DIExpression(), !80)
  %this1 = load ptr, ptr %this.addr, align 8
  %1 = load ptr, ptr %.addr, align 8, !dbg !81
  %call = call noundef ptr @_ZN16DefaultedOutlineC2EOS_(ptr noundef nonnull align 1 dereferenceable(1) %this1, ptr noundef nonnull align 1 dereferenceable(1) %1), !dbg !81
  ret ptr %this1, !dbg !81
}

; Function Attrs: mustprogress noinline nounwind optnone ssp uwtable(sync)
define noundef nonnull align 1 dereferenceable(1) ptr @_ZN16DefaultedOutlineaSEOS_(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef nonnull align 1 dereferenceable(1) %0) #0 !dbg !82 {
entry:
  %this.addr = alloca ptr, align 8
  %.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !83, !DIExpression(), !84)
  store ptr %0, ptr %.addr, align 8
    #dbg_declare(ptr %.addr, !85, !DIExpression(), !86)
  %this1 = load ptr, ptr %this.addr, align 8
  ret ptr %this1, !dbg !87
}

; Function Attrs: mustprogress noinline norecurse optnone ssp uwtable(sync)
define noundef i32 @main() #1 personality ptr @__gxx_personality_v0 !dbg !89 {
entry:
  %retval = alloca i32, align 4
  %a = alloca %class.DefaultedInline, align 1
  %b = alloca %class.DefaultedOutline, align 1
  %c = alloca %class.NeverDefaulted, align 1
  %exn.slot = alloca ptr, align 8
  %ehselector.slot = alloca i32, align 4
  %d = alloca %class.DefaultedInlineWithTemplate, align 1
  %e = alloca %class.DefaultedInlineWithTemplate.0, align 1
  store i32 0, ptr %retval, align 4
    #dbg_declare(ptr %a, !93, !DIExpression(), !117)
    #dbg_declare(ptr %b, !118, !DIExpression(), !119)
  %call = call noundef ptr @_ZN16DefaultedOutlineC1Ev(ptr noundef nonnull align 1 dereferenceable(1) %b), !dbg !119
    #dbg_declare(ptr %c, !120, !DIExpression(), !144)
  %call1 = invoke noundef ptr @_ZN14NeverDefaultedC1Ev(ptr noundef nonnull align 1 dereferenceable(1) %c)
          to label %invoke.cont unwind label %lpad, !dbg !144

invoke.cont:                                      ; preds = %entry
    #dbg_declare(ptr %d, !145, !DIExpression(), !176)
    #dbg_declare(ptr %e, !177, !DIExpression(), !207)
  store i32 0, ptr %retval, align 4, !dbg !208
  %call2 = call noundef ptr @_ZN14NeverDefaultedD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %c) #3, !dbg !209
  %call3 = call noundef ptr @_ZN16DefaultedOutlineD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %b) #3, !dbg !209
  %0 = load i32, ptr %retval, align 4, !dbg !209
  ret i32 %0, !dbg !209

lpad:                                             ; preds = %entry
  %1 = landingpad { ptr, i32 }
          cleanup, !dbg !209
  %2 = extractvalue { ptr, i32 } %1, 0, !dbg !209
  store ptr %2, ptr %exn.slot, align 8, !dbg !209
  %3 = extractvalue { ptr, i32 } %1, 1, !dbg !209
  store i32 %3, ptr %ehselector.slot, align 4, !dbg !209
  %call4 = call noundef ptr @_ZN16DefaultedOutlineD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %b) #3, !dbg !209
  br label %eh.resume, !dbg !209

eh.resume:                                        ; preds = %lpad
  %exn = load ptr, ptr %exn.slot, align 8, !dbg !209
  %sel = load i32, ptr %ehselector.slot, align 4, !dbg !209
  %lpad.val = insertvalue { ptr, i32 } poison, ptr %exn, 0, !dbg !209
  %lpad.val5 = insertvalue { ptr, i32 } %lpad.val, i32 %sel, 1, !dbg !209
  resume { ptr, i32 } %lpad.val5, !dbg !209
}

; Function Attrs: mustprogress noinline optnone ssp uwtable(sync)
define linkonce_odr noundef ptr @_ZN14NeverDefaultedC1Ev(ptr noundef nonnull returned align 1 dereferenceable(1) %this) unnamed_addr #2 !dbg !210 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !211, !DIExpression(), !213)
  %this1 = load ptr, ptr %this.addr, align 8
  %call = call noundef ptr @_ZN14NeverDefaultedC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this1), !dbg !214
  ret ptr %this1, !dbg !215
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress noinline nounwind optnone ssp uwtable(sync)
define linkonce_odr noundef ptr @_ZN14NeverDefaultedD1Ev(ptr noundef nonnull returned align 1 dereferenceable(1) %this) unnamed_addr #0 !dbg !216 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !219, !DIExpression(), !220)
  %this1 = load ptr, ptr %this.addr, align 8
  %call = call noundef ptr @_ZN14NeverDefaultedD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this1) #3, !dbg !221
  ret ptr %this1, !dbg !222
}

; Function Attrs: mustprogress noinline nounwind optnone ssp uwtable(sync)
define linkonce_odr noundef ptr @_ZN14NeverDefaultedC2Ev(ptr noundef nonnull returned align 1 dereferenceable(1) %this) unnamed_addr #0 !dbg !223 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !224, !DIExpression(), !225)
  %this1 = load ptr, ptr %this.addr, align 8
  ret ptr %this1, !dbg !226
}

; Function Attrs: mustprogress noinline nounwind optnone ssp uwtable(sync)
define linkonce_odr noundef ptr @_ZN14NeverDefaultedD2Ev(ptr noundef nonnull returned align 1 dereferenceable(1) %this) unnamed_addr #0 !dbg !227 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !228, !DIExpression(), !229)
  %this1 = load ptr, ptr %this.addr, align 8
  ret ptr %this1, !dbg !230
}

attributes #0 = { mustprogress noinline nounwind optnone ssp uwtable(sync) "frame-pointer"="non-leaf-no-reserve" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+ccpp,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a" "tune-cpu"="apple-m5" }
attributes #1 = { mustprogress noinline norecurse optnone ssp uwtable(sync) "frame-pointer"="non-leaf-no-reserve" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+ccpp,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a" "tune-cpu"="apple-m5" }
attributes #2 = { mustprogress noinline optnone ssp uwtable(sync) "frame-pointer"="non-leaf-no-reserve" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+ccpp,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a" "tune-cpu"="apple-m5" }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 24.0.0git (git@github.com:llvm/llvm-project.git 3919897e1b1b2c4f4327e9a8a9eebfb91bc75616)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
!1 = !DIFile(filename: "debug-info-defaulted.cpp", directory: "/Users/alex/Projects/OSS/llvm-project/build", checksumkind: CSK_MD5, checksum: "7abd2c850fe68e0ff694a41ded15ad61")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 8, !"PIC Level", i32 2}
!5 = !{i32 7, !"uwtable", i32 1}
!6 = !{i32 7, !"frame-pointer", i32 4}
!7 = !{!"clang version 24.0.0git (git@github.com:llvm/llvm-project.git 3919897e1b1b2c4f4327e9a8a9eebfb91bc75616)"}
!8 = distinct !DISubprogram(name: "DefaultedOutline", linkageName: "_ZN16DefaultedOutlineC2Ev", scope: !9, file: !1, line: 25, type: !12, scopeLine: 25, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagDefaultedOutOfClass, unit: !0, declaration: !11, retainedNodes: !32)
!9 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "DefaultedOutline", file: !1, line: 13, size: 8, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !10, identifier: "_ZTS16DefaultedOutline")
!10 = !{!11, !15, !16, !21, !25, !29}
!11 = !DISubprogram(name: "DefaultedOutline", linkageName: "_ZN16DefaultedOutlineC4Ev", scope: !9, file: !1, line: 15, type: !12, scopeLine: 15, flags: DIFlagPublic | DIFlagPrototyped, spFlags: 0)
!12 = !DISubroutineType(types: !13)
!13 = !{null, !14}
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!15 = !DISubprogram(name: "~DefaultedOutline", linkageName: "_ZN16DefaultedOutlineD4Ev", scope: !9, file: !1, line: 16, type: !12, scopeLine: 16, flags: DIFlagPublic | DIFlagPrototyped, spFlags: 0)
!16 = !DISubprogram(name: "DefaultedOutline", linkageName: "_ZN16DefaultedOutlineC4ERKS_", scope: !9, file: !1, line: 18, type: !17, scopeLine: 18, flags: DIFlagPublic | DIFlagPrototyped, spFlags: 0)
!17 = !DISubroutineType(types: !18)
!18 = !{null, !14, !19}
!19 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !20, size: 64)
!20 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !9)
!21 = !DISubprogram(name: "operator=", linkageName: "_ZN16DefaultedOutlineaSERKS_", scope: !9, file: !1, line: 19, type: !22, scopeLine: 19, flags: DIFlagPublic | DIFlagPrototyped, spFlags: 0)
!22 = !DISubroutineType(types: !23)
!23 = !{!24, !14, !19}
!24 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !9, size: 64)
!25 = !DISubprogram(name: "DefaultedOutline", linkageName: "_ZN16DefaultedOutlineC4EOS_", scope: !9, file: !1, line: 21, type: !26, scopeLine: 21, flags: DIFlagPublic | DIFlagPrototyped, spFlags: 0)
!26 = !DISubroutineType(types: !27)
!27 = !{null, !14, !28}
!28 = !DIDerivedType(tag: DW_TAG_rvalue_reference_type, baseType: !9, size: 64)
!29 = !DISubprogram(name: "operator=", linkageName: "_ZN16DefaultedOutlineaSEOS_", scope: !9, file: !1, line: 22, type: !30, scopeLine: 22, flags: DIFlagPublic | DIFlagPrototyped, spFlags: 0)
!30 = !DISubroutineType(types: !31)
!31 = !{!24, !14, !28}
!32 = !{}
!33 = !DILocalVariable(name: "this", arg: 1, scope: !8, type: !34, flags: DIFlagArtificial | DIFlagObjectPointer)
!34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!35 = !DILocation(line: 0, scope: !8)
!36 = !DILocation(line: 25, column: 40, scope: !8)
!37 = distinct !DISubprogram(name: "DefaultedOutline", linkageName: "_ZN16DefaultedOutlineC1Ev", scope: !9, file: !1, line: 25, type: !12, scopeLine: 25, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagDefaultedOutOfClass, unit: !0, declaration: !11, retainedNodes: !32)
!38 = !DILocalVariable(name: "this", arg: 1, scope: !37, type: !34, flags: DIFlagArtificial | DIFlagObjectPointer)
!39 = !DILocation(line: 0, scope: !37)
!40 = !DILocation(line: 25, column: 40, scope: !37)
!41 = distinct !DISubprogram(name: "~DefaultedOutline", linkageName: "_ZN16DefaultedOutlineD2Ev", scope: !9, file: !1, line: 26, type: !42, scopeLine: 26, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagDefaultedOutOfClass, unit: !0, declaration: !15, retainedNodes: !32)
!42 = !DISubroutineType(types: !43)
!43 = !{!34, !14}
!44 = !DILocalVariable(name: "this", arg: 1, scope: !41, type: !34, flags: DIFlagArtificial | DIFlagObjectPointer)
!45 = !DILocation(line: 0, scope: !41)
!46 = !DILocation(line: 26, column: 41, scope: !41)
!47 = distinct !DISubprogram(name: "~DefaultedOutline", linkageName: "_ZN16DefaultedOutlineD1Ev", scope: !9, file: !1, line: 26, type: !42, scopeLine: 26, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagDefaultedOutOfClass, unit: !0, declaration: !15, retainedNodes: !32)
!48 = !DILocalVariable(name: "this", arg: 1, scope: !47, type: !34, flags: DIFlagArtificial | DIFlagObjectPointer)
!49 = !DILocation(line: 0, scope: !47)
!50 = !DILocation(line: 26, column: 41, scope: !47)
!51 = distinct !DISubprogram(name: "DefaultedOutline", linkageName: "_ZN16DefaultedOutlineC2ERKS_", scope: !9, file: !1, line: 28, type: !17, scopeLine: 28, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagDefaultedOutOfClass, unit: !0, declaration: !16, retainedNodes: !32)
!52 = !DILocalVariable(name: "this", arg: 1, scope: !51, type: !34, flags: DIFlagArtificial | DIFlagObjectPointer)
!53 = !DILocation(line: 0, scope: !51)
!54 = !DILocalVariable(arg: 2, scope: !51, file: !1, line: 28, type: !19)
!55 = !DILocation(line: 28, column: 60, scope: !51)
!56 = !DILocation(line: 28, column: 64, scope: !51)
!57 = distinct !DISubprogram(name: "DefaultedOutline", linkageName: "_ZN16DefaultedOutlineC1ERKS_", scope: !9, file: !1, line: 28, type: !17, scopeLine: 28, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagDefaultedOutOfClass, unit: !0, declaration: !16, retainedNodes: !32)
!58 = !DILocalVariable(name: "this", arg: 1, scope: !57, type: !34, flags: DIFlagArtificial | DIFlagObjectPointer)
!59 = !DILocation(line: 0, scope: !57)
!60 = !DILocalVariable(arg: 2, scope: !57, file: !1, line: 28, type: !19)
!61 = !DILocation(line: 28, column: 60, scope: !57)
!62 = !DILocation(line: 28, column: 64, scope: !57)
!63 = distinct !DISubprogram(name: "operator=", linkageName: "_ZN16DefaultedOutlineaSERKS_", scope: !9, file: !1, line: 30, type: !22, scopeLine: 30, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagDefaultedOutOfClass, unit: !0, declaration: !21, retainedNodes: !32)
!64 = !DILocalVariable(name: "this", arg: 1, scope: !63, type: !34, flags: DIFlagArtificial | DIFlagObjectPointer)
!65 = !DILocation(line: 0, scope: !63)
!66 = !DILocalVariable(arg: 2, scope: !63, file: !1, line: 30, type: !19)
!67 = !DILocation(line: 30, column: 53, scope: !63)
!68 = !DILocation(line: 30, column: 57, scope: !69)
!69 = distinct !DILexicalBlock(scope: !63, file: !1, line: 30, column: 57)
!70 = distinct !DISubprogram(name: "DefaultedOutline", linkageName: "_ZN16DefaultedOutlineC2EOS_", scope: !9, file: !1, line: 32, type: !26, scopeLine: 32, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagDefaultedOutOfClass, unit: !0, declaration: !25, retainedNodes: !32)
!71 = !DILocalVariable(name: "this", arg: 1, scope: !70, type: !34, flags: DIFlagArtificial | DIFlagObjectPointer)
!72 = !DILocation(line: 0, scope: !70)
!73 = !DILocalVariable(arg: 2, scope: !70, file: !1, line: 32, type: !28)
!74 = !DILocation(line: 32, column: 55, scope: !70)
!75 = !DILocation(line: 32, column: 59, scope: !70)
!76 = distinct !DISubprogram(name: "DefaultedOutline", linkageName: "_ZN16DefaultedOutlineC1EOS_", scope: !9, file: !1, line: 32, type: !26, scopeLine: 32, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagDefaultedOutOfClass, unit: !0, declaration: !25, retainedNodes: !32)
!77 = !DILocalVariable(name: "this", arg: 1, scope: !76, type: !34, flags: DIFlagArtificial | DIFlagObjectPointer)
!78 = !DILocation(line: 0, scope: !76)
!79 = !DILocalVariable(arg: 2, scope: !76, file: !1, line: 32, type: !28)
!80 = !DILocation(line: 32, column: 55, scope: !76)
!81 = !DILocation(line: 32, column: 59, scope: !76)
!82 = distinct !DISubprogram(name: "operator=", linkageName: "_ZN16DefaultedOutlineaSEOS_", scope: !9, file: !1, line: 33, type: !30, scopeLine: 33, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagDefaultedOutOfClass, unit: !0, declaration: !29, retainedNodes: !32)
!83 = !DILocalVariable(name: "this", arg: 1, scope: !82, type: !34, flags: DIFlagArtificial | DIFlagObjectPointer)
!84 = !DILocation(line: 0, scope: !82)
!85 = !DILocalVariable(arg: 2, scope: !82, file: !1, line: 33, type: !28)
!86 = !DILocation(line: 33, column: 66, scope: !82)
!87 = !DILocation(line: 33, column: 70, scope: !88)
!88 = distinct !DILexicalBlock(scope: !82, file: !1, line: 33, column: 70)
!89 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 63, type: !90, scopeLine: 63, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !32)
!90 = !DISubroutineType(types: !91)
!91 = !{!92}
!92 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!93 = !DILocalVariable(name: "a", scope: !89, file: !1, line: 64, type: !94)
!94 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "DefaultedInline", file: !1, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !95, identifier: "_ZTS15DefaultedInline")
!95 = !{!96, !100, !101, !106, !110, !114}
!96 = !DISubprogram(name: "DefaultedInline", linkageName: "_ZN15DefaultedInlineC4Ev", scope: !94, file: !1, line: 3, type: !97, scopeLine: 3, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDefaultedInClass)
!97 = !DISubroutineType(types: !98)
!98 = !{null, !99}
!99 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !94, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!100 = !DISubprogram(name: "~DefaultedInline", linkageName: "_ZN15DefaultedInlineD4Ev", scope: !94, file: !1, line: 4, type: !97, scopeLine: 4, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDefaultedInClass)
!101 = !DISubprogram(name: "DefaultedInline", linkageName: "_ZN15DefaultedInlineC4ERKS_", scope: !94, file: !1, line: 6, type: !102, scopeLine: 6, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDefaultedInClass)
!102 = !DISubroutineType(types: !103)
!103 = !{null, !99, !104}
!104 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !105, size: 64)
!105 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !94)
!106 = !DISubprogram(name: "operator=", linkageName: "_ZN15DefaultedInlineaSERKS_", scope: !94, file: !1, line: 7, type: !107, scopeLine: 7, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDefaultedInClass)
!107 = !DISubroutineType(types: !108)
!108 = !{!109, !99, !104}
!109 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !94, size: 64)
!110 = !DISubprogram(name: "DefaultedInline", linkageName: "_ZN15DefaultedInlineC4EOS_", scope: !94, file: !1, line: 9, type: !111, scopeLine: 9, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDefaultedInClass)
!111 = !DISubroutineType(types: !112)
!112 = !{null, !99, !113}
!113 = !DIDerivedType(tag: DW_TAG_rvalue_reference_type, baseType: !94, size: 64)
!114 = !DISubprogram(name: "operator=", linkageName: "_ZN15DefaultedInlineaSEOS_", scope: !94, file: !1, line: 10, type: !115, scopeLine: 10, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDefaultedInClass)
!115 = !DISubroutineType(types: !116)
!116 = !{!109, !99, !113}
!117 = !DILocation(line: 64, column: 19, scope: !89)
!118 = !DILocalVariable(name: "b", scope: !89, file: !1, line: 65, type: !9)
!119 = !DILocation(line: 65, column: 20, scope: !89)
!120 = !DILocalVariable(name: "c", scope: !89, file: !1, line: 66, type: !121)
!121 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "NeverDefaulted", file: !1, line: 35, size: 8, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !122, identifier: "_ZTS14NeverDefaulted")
!122 = !{!123, !127, !128, !133, !137, !141}
!123 = !DISubprogram(name: "NeverDefaulted", linkageName: "_ZN14NeverDefaultedC4Ev", scope: !121, file: !1, line: 37, type: !124, scopeLine: 37, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDefaultedNo)
!124 = !DISubroutineType(types: !125)
!125 = !{null, !126}
!126 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !121, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!127 = !DISubprogram(name: "~NeverDefaulted", linkageName: "_ZN14NeverDefaultedD4Ev", scope: !121, file: !1, line: 38, type: !124, scopeLine: 38, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDefaultedNo)
!128 = !DISubprogram(name: "NeverDefaulted", linkageName: "_ZN14NeverDefaultedC4ERKS_", scope: !121, file: !1, line: 40, type: !129, scopeLine: 40, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDefaultedNo)
!129 = !DISubroutineType(types: !130)
!130 = !{null, !126, !131}
!131 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !132, size: 64)
!132 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !121)
!133 = !DISubprogram(name: "operator=", linkageName: "_ZN14NeverDefaultedaSERKS_", scope: !121, file: !1, line: 41, type: !134, scopeLine: 41, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDefaultedNo)
!134 = !DISubroutineType(types: !135)
!135 = !{!136, !126, !131}
!136 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !121, size: 64)
!137 = !DISubprogram(name: "NeverDefaulted", linkageName: "_ZN14NeverDefaultedC4EOS_", scope: !121, file: !1, line: 43, type: !138, scopeLine: 43, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDefaultedNo)
!138 = !DISubroutineType(types: !139)
!139 = !{null, !126, !140}
!140 = !DIDerivedType(tag: DW_TAG_rvalue_reference_type, baseType: !121, size: 64)
!141 = !DISubprogram(name: "operator=", linkageName: "_ZN14NeverDefaultedaSEOS_", scope: !121, file: !1, line: 44, type: !142, scopeLine: 44, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDefaultedNo)
!142 = !DISubroutineType(types: !143)
!143 = !{!136, !126, !140}
!144 = !DILocation(line: 66, column: 18, scope: !89)
!145 = !DILocalVariable(name: "d", scope: !89, file: !1, line: 67, type: !146)
!146 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "DefaultedInlineWithTemplate", file: !1, line: 47, size: 48, flags: DIFlagTypePassByValue | DIFlagNameIsSimplified, elements: !147, templateParams: !174, identifier: "_ZTS27DefaultedInlineWithTemplateILi6EE")
!147 = !{!148, !153, !157, !158, !163, !167, !171}
!148 = !DIDerivedType(tag: DW_TAG_member, name: "m", scope: !146, file: !1, line: 49, baseType: !149, size: 48, flags: DIFlagPublic)
!149 = !DICompositeType(tag: DW_TAG_array_type, baseType: !150, size: 48, elements: !151)
!150 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!151 = !{!152}
!152 = !DISubrange(count: 6)
!153 = !DISubprogram(name: "DefaultedInlineWithTemplate", linkageName: "_ZN27DefaultedInlineWithTemplateILi6EEC4Ev", scope: !146, file: !1, line: 51, type: !154, scopeLine: 51, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDefaultedInClass)
!154 = !DISubroutineType(types: !155)
!155 = !{null, !156}
!156 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !146, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!157 = !DISubprogram(name: "~DefaultedInlineWithTemplate", linkageName: "_ZN27DefaultedInlineWithTemplateILi6EED4Ev", scope: !146, file: !1, line: 52, type: !154, scopeLine: 52, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDefaultedInClass)
!158 = !DISubprogram(name: "DefaultedInlineWithTemplate", linkageName: "_ZN27DefaultedInlineWithTemplateILi6EEC4ERKS0_", scope: !146, file: !1, line: 54, type: !159, scopeLine: 54, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDefaultedInClass)
!159 = !DISubroutineType(types: !160)
!160 = !{null, !156, !161}
!161 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !162, size: 64)
!162 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !146)
!163 = !DISubprogram(name: "operator=", linkageName: "_ZN27DefaultedInlineWithTemplateILi6EEaSERKS0_", scope: !146, file: !1, line: 56, type: !164, scopeLine: 56, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDefaultedInClass)
!164 = !DISubroutineType(types: !165)
!165 = !{!166, !156, !161}
!166 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !146, size: 64)
!167 = !DISubprogram(name: "DefaultedInlineWithTemplate", linkageName: "_ZN27DefaultedInlineWithTemplateILi6EEC4EOS0_", scope: !146, file: !1, line: 58, type: !168, scopeLine: 58, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDefaultedInClass)
!168 = !DISubroutineType(types: !169)
!169 = !{null, !156, !170}
!170 = !DIDerivedType(tag: DW_TAG_rvalue_reference_type, baseType: !146, size: 64)
!171 = !DISubprogram(name: "operator=", linkageName: "_ZN27DefaultedInlineWithTemplateILi6EEaSEOS0_", scope: !146, file: !1, line: 60, type: !172, scopeLine: 60, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDefaultedInClass)
!172 = !DISubroutineType(types: !173)
!173 = !{!166, !156, !170}
!174 = !{!175}
!175 = !DITemplateValueParameter(name: "N", type: !92, value: i32 6)
!176 = !DILocation(line: 67, column: 34, scope: !89)
!177 = !DILocalVariable(name: "e", scope: !89, file: !1, line: 68, type: !178)
!178 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "DefaultedInlineWithTemplate", file: !1, line: 47, size: 56, flags: DIFlagTypePassByValue | DIFlagNameIsSimplified, elements: !179, templateParams: !205, identifier: "_ZTS27DefaultedInlineWithTemplateILi7EE")
!179 = !{!180, !184, !188, !189, !194, !198, !202}
!180 = !DIDerivedType(tag: DW_TAG_member, name: "m", scope: !178, file: !1, line: 49, baseType: !181, size: 56, flags: DIFlagPublic)
!181 = !DICompositeType(tag: DW_TAG_array_type, baseType: !150, size: 56, elements: !182)
!182 = !{!183}
!183 = !DISubrange(count: 7)
!184 = !DISubprogram(name: "DefaultedInlineWithTemplate", linkageName: "_ZN27DefaultedInlineWithTemplateILi7EEC4Ev", scope: !178, file: !1, line: 51, type: !185, scopeLine: 51, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDefaultedInClass)
!185 = !DISubroutineType(types: !186)
!186 = !{null, !187}
!187 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !178, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!188 = !DISubprogram(name: "~DefaultedInlineWithTemplate", linkageName: "_ZN27DefaultedInlineWithTemplateILi7EED4Ev", scope: !178, file: !1, line: 52, type: !185, scopeLine: 52, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDefaultedInClass)
!189 = !DISubprogram(name: "DefaultedInlineWithTemplate", linkageName: "_ZN27DefaultedInlineWithTemplateILi7EEC4ERKS0_", scope: !178, file: !1, line: 54, type: !190, scopeLine: 54, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDefaultedInClass)
!190 = !DISubroutineType(types: !191)
!191 = !{null, !187, !192}
!192 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !193, size: 64)
!193 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !178)
!194 = !DISubprogram(name: "operator=", linkageName: "_ZN27DefaultedInlineWithTemplateILi7EEaSERKS0_", scope: !178, file: !1, line: 56, type: !195, scopeLine: 56, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDefaultedInClass)
!195 = !DISubroutineType(types: !196)
!196 = !{!197, !187, !192}
!197 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !178, size: 64)
!198 = !DISubprogram(name: "DefaultedInlineWithTemplate", linkageName: "_ZN27DefaultedInlineWithTemplateILi7EEC4EOS0_", scope: !178, file: !1, line: 58, type: !199, scopeLine: 58, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDefaultedInClass)
!199 = !DISubroutineType(types: !200)
!200 = !{null, !187, !201}
!201 = !DIDerivedType(tag: DW_TAG_rvalue_reference_type, baseType: !178, size: 64)
!202 = !DISubprogram(name: "operator=", linkageName: "_ZN27DefaultedInlineWithTemplateILi7EEaSEOS0_", scope: !178, file: !1, line: 60, type: !203, scopeLine: 60, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDefaultedInClass)
!203 = !DISubroutineType(types: !204)
!204 = !{!197, !187, !201}
!205 = !{!206}
!206 = !DITemplateValueParameter(name: "N", type: !92, value: i32 7)
!207 = !DILocation(line: 68, column: 34, scope: !89)
!208 = !DILocation(line: 69, column: 3, scope: !89)
!209 = !DILocation(line: 70, column: 1, scope: !89)
!210 = distinct !DISubprogram(name: "NeverDefaulted", linkageName: "_ZN14NeverDefaultedC1Ev", scope: !121, file: !1, line: 37, type: !124, scopeLine: 37, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagDefaultedNo, unit: !0, declaration: !123, retainedNodes: !32)
!211 = !DILocalVariable(name: "this", arg: 1, scope: !210, type: !212, flags: DIFlagArtificial | DIFlagObjectPointer)
!212 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !121, size: 64)
!213 = !DILocation(line: 0, scope: !210)
!214 = !DILocation(line: 37, column: 20, scope: !210)
!215 = !DILocation(line: 37, column: 21, scope: !210)
!216 = distinct !DISubprogram(name: "~NeverDefaulted", linkageName: "_ZN14NeverDefaultedD1Ev", scope: !121, file: !1, line: 38, type: !217, scopeLine: 38, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagDefaultedNo, unit: !0, declaration: !127, retainedNodes: !32)
!217 = !DISubroutineType(types: !218)
!218 = !{!212, !126}
!219 = !DILocalVariable(name: "this", arg: 1, scope: !216, type: !212, flags: DIFlagArtificial | DIFlagObjectPointer)
!220 = !DILocation(line: 0, scope: !216)
!221 = !DILocation(line: 38, column: 21, scope: !216)
!222 = !DILocation(line: 38, column: 22, scope: !216)
!223 = distinct !DISubprogram(name: "NeverDefaulted", linkageName: "_ZN14NeverDefaultedC2Ev", scope: !121, file: !1, line: 37, type: !124, scopeLine: 37, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagDefaultedNo, unit: !0, declaration: !123, retainedNodes: !32)
!224 = !DILocalVariable(name: "this", arg: 1, scope: !223, type: !212, flags: DIFlagArtificial | DIFlagObjectPointer)
!225 = !DILocation(line: 0, scope: !223)
!226 = !DILocation(line: 37, column: 21, scope: !223)
!227 = distinct !DISubprogram(name: "~NeverDefaulted", linkageName: "_ZN14NeverDefaultedD2Ev", scope: !121, file: !1, line: 38, type: !217, scopeLine: 38, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagDefaultedNo, unit: !0, declaration: !127, retainedNodes: !32)
!228 = !DILocalVariable(name: "this", arg: 1, scope: !227, type: !212, flags: DIFlagArtificial | DIFlagObjectPointer)
!229 = !DILocation(line: 0, scope: !227)
!230 = !DILocation(line: 38, column: 22, scope: !227)
