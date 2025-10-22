; ModuleID = 'PointerTest.cpp'
source_filename = "PointerTest.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.43.34808"

%"class.std::basic_ostream" = type { ptr, [4 x i8], i32, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", ptr, ptr, i8 }
%"class.std::ios_base" = type { ptr, i64, i32, i32, i32, i64, i64, ptr, ptr, ptr }
%"class.std::locale::id" = type { i64 }
%rtti.TypeDescriptor26 = type { ptr, ptr, [27 x i8] }
%eh.CatchableType = type { i32, i32, i32, i32, i32, i32, i32 }
%rtti.TypeDescriptor22 = type { ptr, ptr, [23 x i8] }
%rtti.TypeDescriptor23 = type { ptr, ptr, [24 x i8] }
%rtti.TypeDescriptor19 = type { ptr, ptr, [20 x i8] }
%eh.CatchableTypeArray.5 = type { i32, [5 x i32] }
%eh.ThrowInfo = type { i32, i32, i32, i32 }
%"union.std::error_category::_Addr_storage" = type { i64 }
%rtti.CompleteObjectLocator = type { i32, i32, i32, i32, i32, i32 }
%rtti.TypeDescriptor35 = type { ptr, ptr, [36 x i8] }
%rtti.ClassHierarchyDescriptor = type { i32, i32, i32, i32 }
%rtti.BaseClassDescriptor = type { i32, i32, i32, i32, i32, i32, i32 }
%rtti.TypeDescriptor24 = type { ptr, ptr, [25 x i8] }
%rtti.TypeDescriptor30 = type { ptr, ptr, [31 x i8] }
%eh.CatchableTypeArray.3 = type { i32, [3 x i32] }
%rtti.TypeDescriptor20 = type { ptr, ptr, [21 x i8] }
%rtti.TypeDescriptor21 = type { ptr, ptr, [22 x i8] }
%rtti.TypeDescriptor25 = type { ptr, ptr, [26 x i8] }
%rtti.TypeDescriptor18 = type { ptr, ptr, [19 x i8] }
%eh.CatchableTypeArray.2 = type { i32, [2 x i32] }
%rtti.TypeDescriptor73 = type { ptr, ptr, [74 x i8] }
%"class.std::error_code" = type { i32, ptr }
%"class.std::ios_base::failure" = type { %"class.std::system_error" }
%"class.std::system_error" = type { %"class.std::_System_error" }
%"class.std::_System_error" = type { %"class.std::runtime_error", %"class.std::error_code" }
%"class.std::runtime_error" = type { %"class.std::exception" }
%"class.std::exception" = type { ptr, %struct.__std_exception_data }
%struct.__std_exception_data = type { ptr, i8 }
%"class.std::basic_ostream<char>::sentry" = type { %"class.std::basic_ostream<char>::_Sentry_base", i8 }
%"class.std::basic_ostream<char>::_Sentry_base" = type { ptr }
%"class.std::locale" = type { [8 x i8], ptr }
%"class.std::ostreambuf_iterator" = type { i8, ptr }
%"class.std::basic_string" = type { %"class.std::_Compressed_pair" }
%"class.std::_Compressed_pair" = type { %"class.std::_String_val" }
%"class.std::_String_val" = type { %"union.std::_String_val<std::_Simple_types<char>>::_Bxty", i64, i64 }
%"union.std::_String_val<std::_Simple_types<char>>::_Bxty" = type { ptr, [8 x i8] }
%"class.std::error_condition" = type { i32, ptr }
%"class.std::bad_array_new_length" = type { %"class.std::bad_alloc" }
%"class.std::bad_alloc" = type { %"class.std::exception" }
%"class.std::_Locinfo" = type { %"class.std::_Lockit", %"class.std::_Yarn", %"class.std::_Yarn", %"class.std::_Yarn.1", %"class.std::_Yarn.1", %"class.std::_Yarn", %"class.std::_Yarn" }
%"class.std::_Lockit" = type { i32 }
%"class.std::_Yarn.1" = type { ptr, i16 }
%"class.std::_Yarn" = type { ptr, i8 }
%struct._Ctypevec = type { i32, ptr, i32, ptr }
%"class.std::bad_cast" = type { %"class.std::exception" }
%struct._Cvtvec = type { i32, i32, i32, [32 x i8] }
%"struct.std::_Tidy_guard" = type { ptr }

$"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z" = comdat any

$"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z" = comdat any

$"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z" = comdat any

$"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@PEBX@Z" = comdat any

$"??__E?id@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A@@YAXXZ" = comdat any

$"??__E?id@?$numpunct@D@std@@2V0locale@2@A@@YAXXZ" = comdat any

$"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z" = comdat any

$"??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ" = comdat any

$"?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ" = comdat any

$"??1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ" = comdat any

$"?make_error_code@std@@YA?AVerror_code@1@W4io_errc@1@@Z" = comdat any

$"??0failure@ios_base@std@@QEAA@PEBDAEBVerror_code@2@@Z" = comdat any

$"??0failure@ios_base@std@@QEAA@AEBV012@@Z" = comdat any

$"??0system_error@std@@QEAA@AEBV01@@Z" = comdat any

$"??0_System_error@std@@QEAA@AEBV01@@Z" = comdat any

$"??0runtime_error@std@@QEAA@AEBV01@@Z" = comdat any

$"??0exception@std@@QEAA@AEBV01@@Z" = comdat any

$"??_G_Iostream_error_category2@std@@UEAAPEAXI@Z" = comdat any

$"?name@_Iostream_error_category2@std@@UEBAPEBDXZ" = comdat any

$"?message@_Iostream_error_category2@std@@UEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@H@Z" = comdat any

$"?default_error_condition@error_category@std@@UEBA?AVerror_condition@2@H@Z" = comdat any

$"?equivalent@error_category@std@@UEBA_NAEBVerror_code@2@H@Z" = comdat any

$"?equivalent@error_category@std@@UEBA_NHAEBVerror_condition@2@@Z" = comdat any

$"?_Xlen_string@std@@YAXXZ" = comdat any

$"?_Throw_bad_array_new_length@std@@YAXXZ" = comdat any

$"??0bad_array_new_length@std@@QEAA@AEBV01@@Z" = comdat any

$"??0bad_alloc@std@@QEAA@AEBV01@@Z" = comdat any

$"??_Gbad_array_new_length@std@@UEAAPEAXI@Z" = comdat any

$"?what@exception@std@@UEBAPEBDXZ" = comdat any

$"??_Gbad_alloc@std@@UEAAPEAXI@Z" = comdat any

$"??_Gexception@std@@UEAAPEAXI@Z" = comdat any

$"??0system_error@std@@QEAA@Verror_code@1@PEBD@Z" = comdat any

$"??_Gfailure@ios_base@std@@UEAAPEAXI@Z" = comdat any

$"??0_System_error@std@@IEAA@Verror_code@1@AEBV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@1@@Z" = comdat any

$"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ" = comdat any

$"??_Gsystem_error@std@@UEAAPEAXI@Z" = comdat any

$"?_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z" = comdat any

$"??_G_System_error@std@@UEAAPEAXI@Z" = comdat any

$"??$_Reallocate_grow_by@V<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@QEBD_K@Z@PEBD_K@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??append@01@QEAAAEAV01@QEBD0@Z@PEBD_K@Z" = comdat any

$"??_Gruntime_error@std@@UEAAPEAXI@Z" = comdat any

$"?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ" = comdat any

$"?put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z" = comdat any

$"??$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z" = comdat any

$"?_Throw_bad_cast@std@@YAXXZ" = comdat any

$"??0_Locinfo@std@@QEAA@PEBD@Z" = comdat any

$"??1_Locinfo@std@@QEAA@XZ" = comdat any

$"??_G?$ctype@D@std@@MEAAPEAXI@Z" = comdat any

$"?_Incref@facet@locale@std@@UEAAXXZ" = comdat any

$"?_Decref@facet@locale@std@@UEAAPEAV_Facet_base@3@XZ" = comdat any

$"?do_tolower@?$ctype@D@std@@MEBAPEBDPEADPEBD@Z" = comdat any

$"?do_tolower@?$ctype@D@std@@MEBADD@Z" = comdat any

$"?do_toupper@?$ctype@D@std@@MEBAPEBDPEADPEBD@Z" = comdat any

$"?do_toupper@?$ctype@D@std@@MEBADD@Z" = comdat any

$"?do_widen@?$ctype@D@std@@MEBAPEBDPEBD0PEAD@Z" = comdat any

$"?do_widen@?$ctype@D@std@@MEBADD@Z" = comdat any

$"?do_narrow@?$ctype@D@std@@MEBAPEBDPEBD0DPEAD@Z" = comdat any

$"?do_narrow@?$ctype@D@std@@MEBADDD@Z" = comdat any

$"??_Gctype_base@std@@UEAAPEAXI@Z" = comdat any

$"??_Gfacet@locale@std@@MEAAPEAXI@Z" = comdat any

$"??_G_Facet_base@std@@UEAAPEAXI@Z" = comdat any

$"??0bad_cast@std@@QEAA@AEBV01@@Z" = comdat any

$"??1exception@std@@UEAA@XZ" = comdat any

$"??_Gbad_cast@std@@UEAAPEAXI@Z" = comdat any

$"??$use_facet@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z" = comdat any

$"??_G?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAAPEAXI@Z" = comdat any

$"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBX@Z" = comdat any

$"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DO@Z" = comdat any

$"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DN@Z" = comdat any

$"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_K@Z" = comdat any

$"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_J@Z" = comdat any

$"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DK@Z" = comdat any

$"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DJ@Z" = comdat any

$"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_N@Z" = comdat any

$"?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z" = comdat any

$sprintf_s = comdat any

$"??$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z" = comdat any

$"?_Getcat@?$numpunct@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z" = comdat any

$"?_Init@?$numpunct@D@std@@IEAAXAEBV_Locinfo@2@_N@Z" = comdat any

$"??_G?$numpunct@D@std@@MEAAPEAXI@Z" = comdat any

$"?do_decimal_point@?$numpunct@D@std@@MEBADXZ" = comdat any

$"?do_thousands_sep@?$numpunct@D@std@@MEBADXZ" = comdat any

$"?do_grouping@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ" = comdat any

$"?do_falsename@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ" = comdat any

$"?do_truename@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ" = comdat any

$"??1?$_Tidy_guard@V?$numpunct@D@std@@@std@@QEAA@XZ" = comdat any

$"??$_Reallocate_grow_by@V<lambda_1>@?0??insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@_K0D@Z@_K_KD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??insert@01@QEAAAEAV01@00D@Z@_K2D@Z" = comdat any

$"?_Xran@?$_String_val@U?$_Simple_types@D@std@@@std@@SAXXZ" = comdat any

$__local_stdio_printf_options = comdat any

$"??$_Fput_v3@$0A@@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@1@V21@AEAVios_base@1@DPEBD_K_N@Z" = comdat any

$"??$_Reallocate_grow_by@V<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@_KD@Z@_KD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??append@01@QEAAAEAV01@0D@Z@_KD@Z" = comdat any

$"??_C@_0BN@GIAMKINO@1?4?5Basic?5pointer?5operations?3?$AA@" = comdat any

$"??_C@_0P@PPPILKBG@Value?5of?5num?3?5?$AA@" = comdat any

$"??_C@_0BB@IGENIPIM@Address?5of?5num?3?5?$AA@" = comdat any

$"??_C@_0P@PDDOLHIE@Value?5of?5ptr?3?5?$AA@" = comdat any

$"??_C@_0BH@DIAOJILP@Value?5pointed?5by?5ptr?3?5?$AA@" = comdat any

$"??_C@_0CJ@OJKBMIMM@2?4?5After?5dereferencing?5and?5modif@" = comdat any

$"??_C@_0BD@PDLLNEEM@New?5value?5of?5num?3?5?$AA@" = comdat any

$"??_C@_0CC@HLHDMDK@3?4?5Pointer?5arithmetic?5with?5array@" = comdat any

$"??_C@_0BA@COOFDBFC@First?5element?3?5?$AA@" = comdat any

$"??_C@_04LNPKFDKO@?5at?5?$AA@" = comdat any

$"??_C@_0BB@GFLPPJEB@Second?5element?3?5?$AA@" = comdat any

$"??_C@_0BA@OJNMCLP@Third?5element?3?5?$AA@" = comdat any

$"??_C@_0BO@LFEDEBBC@4?4?5Dynamic?5memory?5allocation?3?$AA@" = comdat any

$"??_C@_0BO@NBEBKJHP@Dynamically?5allocated?5value?3?5?$AA@" = comdat any

$"??_C@_0BH@LDGOONGC@5?4?5Pointers?5and?5const?3?$AA@" = comdat any

$"??_C@_0CB@GFNKDNIK@Constant?5value?5through?5pointer?3?5@" = comdat any

$"??_C@_0BG@OMIOAIGA@6?4?5Array?5of?5pointers?3?$AA@" = comdat any

$"??_C@_08IMOEKGIO@Pointer?5?$AA@" = comdat any

$"??_C@_0BD@INFLFJBG@?5points?5to?5value?3?5?$AA@" = comdat any

$"??_C@_0BH@IEOBKIOK@7?4?5Pointer?5comparison?3?$AA@" = comdat any

$"??_C@_0BH@DGOHBFGE@ptr?5is?5pointing?5to?5num?$AA@" = comdat any

$"??_C@_0CB@ILBHDHPI@arrPtr?5is?5ahead?5of?5arr?5in?5memory@" = comdat any

$"?id@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A" = comdat any

$"?id@?$numpunct@D@std@@2V0locale@2@A" = comdat any

$"??_C@_0BF@PHHKMMFD@ios_base?3?3badbit?5set?$AA@" = comdat any

$"??_C@_0BG@FMKFHCIL@ios_base?3?3failbit?5set?$AA@" = comdat any

$"??_C@_0BF@OOHOMBOF@ios_base?3?3eofbit?5set?$AA@" = comdat any

$"??_R0?AVfailure@ios_base@std@@@8" = comdat any

$"_CT??_R0?AVfailure@ios_base@std@@@8??0failure@ios_base@std@@QEAA@AEBV012@@Z40" = comdat any

$"??_R0?AVsystem_error@std@@@8" = comdat any

$"_CT??_R0?AVsystem_error@std@@@8??0system_error@std@@QEAA@AEBV01@@Z40" = comdat any

$"??_R0?AV_System_error@std@@@8" = comdat any

$"_CT??_R0?AV_System_error@std@@@8??0_System_error@std@@QEAA@AEBV01@@Z40" = comdat any

$"??_R0?AVruntime_error@std@@@8" = comdat any

$"_CT??_R0?AVruntime_error@std@@@8??0runtime_error@std@@QEAA@AEBV01@@Z24" = comdat any

$"??_R0?AVexception@std@@@8" = comdat any

$"_CT??_R0?AVexception@std@@@8??0exception@std@@QEAA@AEBV01@@Z24" = comdat any

$"_CTA5?AVfailure@ios_base@std@@" = comdat any

$"_TI5?AVfailure@ios_base@std@@" = comdat any

$"?_Static@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@1@XZ@4V21@A" = comdat any

$"??_7_Iostream_error_category2@std@@6B@" = comdat largest

$"?$TSS0@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@1@XZ@4HA" = comdat any

$"??_R4_Iostream_error_category2@std@@6B@" = comdat any

$"??_R0?AV_Iostream_error_category2@std@@@8" = comdat any

$"??_R3_Iostream_error_category2@std@@8" = comdat any

$"??_R2_Iostream_error_category2@std@@8" = comdat any

$"??_R1A@?0A@EA@_Iostream_error_category2@std@@8" = comdat any

$"??_R1A@?0A@EA@error_category@std@@8" = comdat any

$"??_R0?AVerror_category@std@@@8" = comdat any

$"??_R3error_category@std@@8" = comdat any

$"??_R2error_category@std@@8" = comdat any

$"??_C@_08LLGCOLLL@iostream?$AA@" = comdat any

$"?_Iostream_error@?4??message@_Iostream_error_category2@std@@UEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@3@H@Z@4QBDB" = comdat any

$"??_C@_0BA@JFNIOLAK@string?5too?5long?$AA@" = comdat any

$"??_R0?AVbad_array_new_length@std@@@8" = comdat any

$"_CT??_R0?AVbad_array_new_length@std@@@8??0bad_array_new_length@std@@QEAA@AEBV01@@Z24" = comdat any

$"??_R0?AVbad_alloc@std@@@8" = comdat any

$"_CT??_R0?AVbad_alloc@std@@@8??0bad_alloc@std@@QEAA@AEBV01@@Z24" = comdat any

$"_CTA3?AVbad_array_new_length@std@@" = comdat any

$"_TI3?AVbad_array_new_length@std@@" = comdat any

$"??_C@_0BF@KINCDENJ@bad?5array?5new?5length?$AA@" = comdat any

$"??_7bad_array_new_length@std@@6B@" = comdat largest

$"??_R4bad_array_new_length@std@@6B@" = comdat any

$"??_R3bad_array_new_length@std@@8" = comdat any

$"??_R2bad_array_new_length@std@@8" = comdat any

$"??_R1A@?0A@EA@bad_array_new_length@std@@8" = comdat any

$"??_R1A@?0A@EA@bad_alloc@std@@8" = comdat any

$"??_R3bad_alloc@std@@8" = comdat any

$"??_R2bad_alloc@std@@8" = comdat any

$"??_R1A@?0A@EA@exception@std@@8" = comdat any

$"??_R3exception@std@@8" = comdat any

$"??_R2exception@std@@8" = comdat any

$"??_7bad_alloc@std@@6B@" = comdat largest

$"??_R4bad_alloc@std@@6B@" = comdat any

$"??_7exception@std@@6B@" = comdat largest

$"??_R4exception@std@@6B@" = comdat any

$"??_C@_0BC@EOODALEL@Unknown?5exception?$AA@" = comdat any

$"??_7failure@ios_base@std@@6B@" = comdat largest

$"??_R4failure@ios_base@std@@6B@" = comdat any

$"??_R3failure@ios_base@std@@8" = comdat any

$"??_R2failure@ios_base@std@@8" = comdat any

$"??_R1A@?0A@EA@failure@ios_base@std@@8" = comdat any

$"??_R1A@?0A@EA@system_error@std@@8" = comdat any

$"??_R3system_error@std@@8" = comdat any

$"??_R2system_error@std@@8" = comdat any

$"??_R1A@?0A@EA@_System_error@std@@8" = comdat any

$"??_R3_System_error@std@@8" = comdat any

$"??_R2_System_error@std@@8" = comdat any

$"??_R1A@?0A@EA@runtime_error@std@@8" = comdat any

$"??_R3runtime_error@std@@8" = comdat any

$"??_R2runtime_error@std@@8" = comdat any

$"??_7system_error@std@@6B@" = comdat largest

$"??_R4system_error@std@@6B@" = comdat any

$"??_7_System_error@std@@6B@" = comdat largest

$"??_R4_System_error@std@@6B@" = comdat any

$"??_C@_02LMMGGCAJ@?3?5?$AA@" = comdat any

$"??_7runtime_error@std@@6B@" = comdat largest

$"??_R4runtime_error@std@@6B@" = comdat any

$"?_Psave@?$_Facetptr@V?$ctype@D@std@@@std@@2PEBVfacet@locale@2@EB" = comdat any

$"??_C@_00CNPNBAHC@?$AA@" = comdat any

$"??_C@_0BA@ELKIONDK@bad?5locale?5name?$AA@" = comdat any

$"??_7?$ctype@D@std@@6B@" = comdat largest

$"??_R4?$ctype@D@std@@6B@" = comdat any

$"??_R0?AV?$ctype@D@std@@@8" = comdat any

$"??_R3?$ctype@D@std@@8" = comdat any

$"??_R2?$ctype@D@std@@8" = comdat any

$"??_R1A@?0A@EA@?$ctype@D@std@@8" = comdat any

$"??_R1A@?0A@EA@ctype_base@std@@8" = comdat any

$"??_R0?AUctype_base@std@@@8" = comdat any

$"??_R3ctype_base@std@@8" = comdat any

$"??_R2ctype_base@std@@8" = comdat any

$"??_R1A@?0A@EA@facet@locale@std@@8" = comdat any

$"??_R0?AVfacet@locale@std@@@8" = comdat any

$"??_R3facet@locale@std@@8" = comdat any

$"??_R2facet@locale@std@@8" = comdat any

$"??_R1A@?0A@EA@_Facet_base@std@@8" = comdat any

$"??_R0?AV_Facet_base@std@@@8" = comdat any

$"??_R3_Facet_base@std@@8" = comdat any

$"??_R2_Facet_base@std@@8" = comdat any

$"??_R17?0A@EA@_Crt_new_delete@std@@8" = comdat any

$"??_R0?AU_Crt_new_delete@std@@@8" = comdat any

$"??_R3_Crt_new_delete@std@@8" = comdat any

$"??_R2_Crt_new_delete@std@@8" = comdat any

$"??_R1A@?0A@EA@_Crt_new_delete@std@@8" = comdat any

$"??_7ctype_base@std@@6B@" = comdat largest

$"??_R4ctype_base@std@@6B@" = comdat any

$"??_7facet@locale@std@@6B@" = comdat largest

$"??_R4facet@locale@std@@6B@" = comdat any

$"??_7_Facet_base@std@@6B@" = comdat largest

$"??_R4_Facet_base@std@@6B@" = comdat any

$"??_R0?AVbad_cast@std@@@8" = comdat any

$"_CT??_R0?AVbad_cast@std@@@8??0bad_cast@std@@QEAA@AEBV01@@Z24" = comdat any

$"_CTA2?AVbad_cast@std@@" = comdat any

$"_TI2?AVbad_cast@std@@" = comdat any

$"??_C@_08EPJLHIJG@bad?5cast?$AA@" = comdat any

$"??_7bad_cast@std@@6B@" = comdat largest

$"??_R4bad_cast@std@@6B@" = comdat any

$"??_R3bad_cast@std@@8" = comdat any

$"??_R2bad_cast@std@@8" = comdat any

$"??_R1A@?0A@EA@bad_cast@std@@8" = comdat any

$"?_Psave@?$_Facetptr@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@2PEBVfacet@locale@2@EB" = comdat any

$"??_7?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@6B@" = comdat largest

$"??_R4?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@6B@" = comdat any

$"??_R0?AV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@8" = comdat any

$"??_R3?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8" = comdat any

$"??_R2?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8" = comdat any

$"??_R1A@?0A@EA@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8" = comdat any

$"??_C@_02BBAHNLBA@?$CFp?$AA@" = comdat any

$"?_Psave@?$_Facetptr@V?$numpunct@D@std@@@std@@2PEBVfacet@locale@2@EB" = comdat any

$"??_7?$numpunct@D@std@@6B@" = comdat largest

$"??_R4?$numpunct@D@std@@6B@" = comdat any

$"??_R0?AV?$numpunct@D@std@@@8" = comdat any

$"??_R3?$numpunct@D@std@@8" = comdat any

$"??_R2?$numpunct@D@std@@8" = comdat any

$"??_R1A@?0A@EA@?$numpunct@D@std@@8" = comdat any

$"??_C@_05LAPONLG@false?$AA@" = comdat any

$"??_C@_04LOAJBDKD@true?$AA@" = comdat any

$"??_C@_0BI@CFPLBAOH@invalid?5string?5position?$AA@" = comdat any

$"?_OptionsStorage@?1??__local_stdio_printf_options@@9@4_KA" = comdat any

$"??_C@_02MDKMJEGG@eE?$AA@" = comdat any

$"??_C@_02OOPEBDOJ@pP?$AA@" = comdat any

@"?cout@std@@3V?$basic_ostream@DU?$char_traits@D@std@@@1@A" = external dso_local global %"class.std::basic_ostream", align 8
@"??_C@_0BN@GIAMKINO@1?4?5Basic?5pointer?5operations?3?$AA@" = linkonce_odr dso_local unnamed_addr constant [29 x i8] c"1. Basic pointer operations:\00", comdat, align 1
@"??_C@_0P@PPPILKBG@Value?5of?5num?3?5?$AA@" = linkonce_odr dso_local unnamed_addr constant [15 x i8] c"Value of num: \00", comdat, align 1
@"??_C@_0BB@IGENIPIM@Address?5of?5num?3?5?$AA@" = linkonce_odr dso_local unnamed_addr constant [17 x i8] c"Address of num: \00", comdat, align 1
@"??_C@_0P@PDDOLHIE@Value?5of?5ptr?3?5?$AA@" = linkonce_odr dso_local unnamed_addr constant [15 x i8] c"Value of ptr: \00", comdat, align 1
@"??_C@_0BH@DIAOJILP@Value?5pointed?5by?5ptr?3?5?$AA@" = linkonce_odr dso_local unnamed_addr constant [23 x i8] c"Value pointed by ptr: \00", comdat, align 1
@"??_C@_0CJ@OJKBMIMM@2?4?5After?5dereferencing?5and?5modif@" = linkonce_odr dso_local unnamed_addr constant [41 x i8] c"2. After dereferencing and modification:\00", comdat, align 1
@"??_C@_0BD@PDLLNEEM@New?5value?5of?5num?3?5?$AA@" = linkonce_odr dso_local unnamed_addr constant [19 x i8] c"New value of num: \00", comdat, align 1
@__const.main.arr = private unnamed_addr constant [5 x i32] [i32 10, i32 20, i32 30, i32 40, i32 50], align 16
@"??_C@_0CC@HLHDMDK@3?4?5Pointer?5arithmetic?5with?5array@" = linkonce_odr dso_local unnamed_addr constant [34 x i8] c"3. Pointer arithmetic with array:\00", comdat, align 1
@"??_C@_0BA@COOFDBFC@First?5element?3?5?$AA@" = linkonce_odr dso_local unnamed_addr constant [16 x i8] c"First element: \00", comdat, align 1
@"??_C@_04LNPKFDKO@?5at?5?$AA@" = linkonce_odr dso_local unnamed_addr constant [5 x i8] c" at \00", comdat, align 1
@"??_C@_0BB@GFLPPJEB@Second?5element?3?5?$AA@" = linkonce_odr dso_local unnamed_addr constant [17 x i8] c"Second element: \00", comdat, align 1
@"??_C@_0BA@OJNMCLP@Third?5element?3?5?$AA@" = linkonce_odr dso_local unnamed_addr constant [16 x i8] c"Third element: \00", comdat, align 1
@"??_C@_0BO@LFEDEBBC@4?4?5Dynamic?5memory?5allocation?3?$AA@" = linkonce_odr dso_local unnamed_addr constant [30 x i8] c"4. Dynamic memory allocation:\00", comdat, align 1
@"??_C@_0BO@NBEBKJHP@Dynamically?5allocated?5value?3?5?$AA@" = linkonce_odr dso_local unnamed_addr constant [30 x i8] c"Dynamically allocated value: \00", comdat, align 1
@"??_C@_0BH@LDGOONGC@5?4?5Pointers?5and?5const?3?$AA@" = linkonce_odr dso_local unnamed_addr constant [23 x i8] c"5. Pointers and const:\00", comdat, align 1
@"??_C@_0CB@GFNKDNIK@Constant?5value?5through?5pointer?3?5@" = linkonce_odr dso_local unnamed_addr constant [33 x i8] c"Constant value through pointer: \00", comdat, align 1
@"??_C@_0BG@OMIOAIGA@6?4?5Array?5of?5pointers?3?$AA@" = linkonce_odr dso_local unnamed_addr constant [22 x i8] c"6. Array of pointers:\00", comdat, align 1
@"??_C@_08IMOEKGIO@Pointer?5?$AA@" = linkonce_odr dso_local unnamed_addr constant [9 x i8] c"Pointer \00", comdat, align 1
@"??_C@_0BD@INFLFJBG@?5points?5to?5value?3?5?$AA@" = linkonce_odr dso_local unnamed_addr constant [19 x i8] c" points to value: \00", comdat, align 1
@"??_C@_0BH@IEOBKIOK@7?4?5Pointer?5comparison?3?$AA@" = linkonce_odr dso_local unnamed_addr constant [23 x i8] c"7. Pointer comparison:\00", comdat, align 1
@"??_C@_0BH@DGOHBFGE@ptr?5is?5pointing?5to?5num?$AA@" = linkonce_odr dso_local unnamed_addr constant [23 x i8] c"ptr is pointing to num\00", comdat, align 1
@"??_C@_0CB@ILBHDHPI@arrPtr?5is?5ahead?5of?5arr?5in?5memory@" = linkonce_odr dso_local unnamed_addr constant [33 x i8] c"arrPtr is ahead of arr in memory\00", comdat, align 1
@"?id@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A" = linkonce_odr dso_local global %"class.std::locale::id" zeroinitializer, comdat, align 8
@"?id@?$numpunct@D@std@@2V0locale@2@A" = linkonce_odr dso_local global %"class.std::locale::id" zeroinitializer, comdat, align 8
@"??_C@_0BF@PHHKMMFD@ios_base?3?3badbit?5set?$AA@" = linkonce_odr dso_local unnamed_addr constant [21 x i8] c"ios_base::badbit set\00", comdat, align 1
@"??_C@_0BG@FMKFHCIL@ios_base?3?3failbit?5set?$AA@" = linkonce_odr dso_local unnamed_addr constant [22 x i8] c"ios_base::failbit set\00", comdat, align 1
@"??_C@_0BF@OOHOMBOF@ios_base?3?3eofbit?5set?$AA@" = linkonce_odr dso_local unnamed_addr constant [21 x i8] c"ios_base::eofbit set\00", comdat, align 1
@"??_7type_info@@6B@" = external constant ptr
@"??_R0?AVfailure@ios_base@std@@@8" = linkonce_odr global %rtti.TypeDescriptor26 { ptr @"??_7type_info@@6B@", ptr null, [27 x i8] c".?AVfailure@ios_base@std@@\00" }, comdat
@__ImageBase = external dso_local constant i8
@"_CT??_R0?AVfailure@ios_base@std@@@8??0failure@ios_base@std@@QEAA@AEBV012@@Z40" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVfailure@ios_base@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 -1, i32 0, i32 40, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??0failure@ios_base@std@@QEAA@AEBV012@@Z" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, section ".xdata", comdat
@"??_R0?AVsystem_error@std@@@8" = linkonce_odr global %rtti.TypeDescriptor22 { ptr @"??_7type_info@@6B@", ptr null, [23 x i8] c".?AVsystem_error@std@@\00" }, comdat
@"_CT??_R0?AVsystem_error@std@@@8??0system_error@std@@QEAA@AEBV01@@Z40" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVsystem_error@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 -1, i32 0, i32 40, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??0system_error@std@@QEAA@AEBV01@@Z" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, section ".xdata", comdat
@"??_R0?AV_System_error@std@@@8" = linkonce_odr global %rtti.TypeDescriptor23 { ptr @"??_7type_info@@6B@", ptr null, [24 x i8] c".?AV_System_error@std@@\00" }, comdat
@"_CT??_R0?AV_System_error@std@@@8??0_System_error@std@@QEAA@AEBV01@@Z40" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AV_System_error@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 -1, i32 0, i32 40, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??0_System_error@std@@QEAA@AEBV01@@Z" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, section ".xdata", comdat
@"??_R0?AVruntime_error@std@@@8" = linkonce_odr global %rtti.TypeDescriptor23 { ptr @"??_7type_info@@6B@", ptr null, [24 x i8] c".?AVruntime_error@std@@\00" }, comdat
@"_CT??_R0?AVruntime_error@std@@@8??0runtime_error@std@@QEAA@AEBV01@@Z24" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVruntime_error@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 -1, i32 0, i32 24, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??0runtime_error@std@@QEAA@AEBV01@@Z" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, section ".xdata", comdat
@"??_R0?AVexception@std@@@8" = linkonce_odr global %rtti.TypeDescriptor19 { ptr @"??_7type_info@@6B@", ptr null, [20 x i8] c".?AVexception@std@@\00" }, comdat
@"_CT??_R0?AVexception@std@@@8??0exception@std@@QEAA@AEBV01@@Z24" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVexception@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 -1, i32 0, i32 24, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??0exception@std@@QEAA@AEBV01@@Z" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, section ".xdata", comdat
@"_CTA5?AVfailure@ios_base@std@@" = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.5 { i32 5, [5 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"_CT??_R0?AVfailure@ios_base@std@@@8??0failure@ios_base@std@@QEAA@AEBV012@@Z40" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"_CT??_R0?AVsystem_error@std@@@8??0system_error@std@@QEAA@AEBV01@@Z40" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"_CT??_R0?AV_System_error@std@@@8??0_System_error@std@@QEAA@AEBV01@@Z40" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"_CT??_R0?AVruntime_error@std@@@8??0runtime_error@std@@QEAA@AEBV01@@Z24" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"_CT??_R0?AVexception@std@@@8??0exception@std@@QEAA@AEBV01@@Z24" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32)] }, section ".xdata", comdat
@"_TI5?AVfailure@ios_base@std@@" = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??1exception@std@@UEAA@XZ" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"_CTA5?AVfailure@ios_base@std@@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, section ".xdata", comdat
@"?_Static@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@1@XZ@4V21@A" = linkonce_odr dso_local global { ptr, %"union.std::error_category::_Addr_storage" } { ptr @"??_7_Iostream_error_category2@std@@6B@", %"union.std::error_category::_Addr_storage" { i64 5 } }, comdat, align 8
@0 = private unnamed_addr constant { [7 x ptr] } { [7 x ptr] [ptr @"??_R4_Iostream_error_category2@std@@6B@", ptr @"??_G_Iostream_error_category2@std@@UEAAPEAXI@Z", ptr @"?name@_Iostream_error_category2@std@@UEBAPEBDXZ", ptr @"?message@_Iostream_error_category2@std@@UEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@H@Z", ptr @"?default_error_condition@error_category@std@@UEBA?AVerror_condition@2@H@Z", ptr @"?equivalent@error_category@std@@UEBA_NAEBVerror_code@2@H@Z", ptr @"?equivalent@error_category@std@@UEBA_NHAEBVerror_condition@2@@Z"] }, comdat($"??_7_Iostream_error_category2@std@@6B@")
@"?$TSS0@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@1@XZ@4HA" = linkonce_odr global i32 0, comdat, align 4
@_Init_thread_epoch = external thread_local local_unnamed_addr global i32, align 4
@"??_R4_Iostream_error_category2@std@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AV_Iostream_error_category2@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3_Iostream_error_category2@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R4_Iostream_error_category2@std@@6B@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R0?AV_Iostream_error_category2@std@@@8" = linkonce_odr global %rtti.TypeDescriptor35 { ptr @"??_7type_info@@6B@", ptr null, [36 x i8] c".?AV_Iostream_error_category2@std@@\00" }, comdat
@"??_R3_Iostream_error_category2@std@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 0, i32 2, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R2_Iostream_error_category2@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R2_Iostream_error_category2@std@@8" = linkonce_odr constant [3 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@_Iostream_error_category2@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@error_category@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0], comdat
@"??_R1A@?0A@EA@_Iostream_error_category2@std@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AV_Iostream_error_category2@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 1, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3_Iostream_error_category2@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R1A@?0A@EA@error_category@std@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVerror_category@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3error_category@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R0?AVerror_category@std@@@8" = linkonce_odr global %rtti.TypeDescriptor24 { ptr @"??_7type_info@@6B@", ptr null, [25 x i8] c".?AVerror_category@std@@\00" }, comdat
@"??_R3error_category@std@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 0, i32 1, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R2error_category@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R2error_category@std@@8" = linkonce_odr constant [2 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@error_category@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0], comdat
@"??_C@_08LLGCOLLL@iostream?$AA@" = linkonce_odr dso_local unnamed_addr constant [9 x i8] c"iostream\00", comdat, align 1
@"?_Iostream_error@?4??message@_Iostream_error_category2@std@@UEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@3@H@Z@4QBDB" = linkonce_odr dso_local local_unnamed_addr constant [22 x i8] c"iostream stream error\00", comdat, align 16
@"??_C@_0BA@JFNIOLAK@string?5too?5long?$AA@" = linkonce_odr dso_local unnamed_addr constant [16 x i8] c"string too long\00", comdat, align 1
@"??_R0?AVbad_array_new_length@std@@@8" = linkonce_odr global %rtti.TypeDescriptor30 { ptr @"??_7type_info@@6B@", ptr null, [31 x i8] c".?AVbad_array_new_length@std@@\00" }, comdat
@"_CT??_R0?AVbad_array_new_length@std@@@8??0bad_array_new_length@std@@QEAA@AEBV01@@Z24" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVbad_array_new_length@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 -1, i32 0, i32 24, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??0bad_array_new_length@std@@QEAA@AEBV01@@Z" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, section ".xdata", comdat
@"??_R0?AVbad_alloc@std@@@8" = linkonce_odr global %rtti.TypeDescriptor19 { ptr @"??_7type_info@@6B@", ptr null, [20 x i8] c".?AVbad_alloc@std@@\00" }, comdat
@"_CT??_R0?AVbad_alloc@std@@@8??0bad_alloc@std@@QEAA@AEBV01@@Z24" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 16, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVbad_alloc@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 -1, i32 0, i32 24, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??0bad_alloc@std@@QEAA@AEBV01@@Z" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, section ".xdata", comdat
@"_CTA3?AVbad_array_new_length@std@@" = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.3 { i32 3, [3 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"_CT??_R0?AVbad_array_new_length@std@@@8??0bad_array_new_length@std@@QEAA@AEBV01@@Z24" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"_CT??_R0?AVbad_alloc@std@@@8??0bad_alloc@std@@QEAA@AEBV01@@Z24" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"_CT??_R0?AVexception@std@@@8??0exception@std@@QEAA@AEBV01@@Z24" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32)] }, section ".xdata", comdat
@"_TI3?AVbad_array_new_length@std@@" = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??1exception@std@@UEAA@XZ" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"_CTA3?AVbad_array_new_length@std@@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, section ".xdata", comdat
@"??_C@_0BF@KINCDENJ@bad?5array?5new?5length?$AA@" = linkonce_odr dso_local unnamed_addr constant [21 x i8] c"bad array new length\00", comdat, align 1
@1 = private unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr @"??_R4bad_array_new_length@std@@6B@", ptr @"??_Gbad_array_new_length@std@@UEAAPEAXI@Z", ptr @"?what@exception@std@@UEBAPEBDXZ"] }, comdat($"??_7bad_array_new_length@std@@6B@")
@"??_R4bad_array_new_length@std@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVbad_array_new_length@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3bad_array_new_length@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R4bad_array_new_length@std@@6B@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R3bad_array_new_length@std@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 0, i32 3, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R2bad_array_new_length@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R2bad_array_new_length@std@@8" = linkonce_odr constant [4 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@bad_array_new_length@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@bad_alloc@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@exception@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0], comdat
@"??_R1A@?0A@EA@bad_array_new_length@std@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVbad_array_new_length@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 2, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3bad_array_new_length@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R1A@?0A@EA@bad_alloc@std@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVbad_alloc@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 1, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3bad_alloc@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R3bad_alloc@std@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 0, i32 2, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R2bad_alloc@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R2bad_alloc@std@@8" = linkonce_odr constant [3 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@bad_alloc@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@exception@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0], comdat
@"??_R1A@?0A@EA@exception@std@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVexception@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3exception@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R3exception@std@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 0, i32 1, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R2exception@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R2exception@std@@8" = linkonce_odr constant [2 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@exception@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0], comdat
@2 = private unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr @"??_R4bad_alloc@std@@6B@", ptr @"??_Gbad_alloc@std@@UEAAPEAXI@Z", ptr @"?what@exception@std@@UEBAPEBDXZ"] }, comdat($"??_7bad_alloc@std@@6B@")
@"??_R4bad_alloc@std@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVbad_alloc@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3bad_alloc@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R4bad_alloc@std@@6B@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@3 = private unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr @"??_R4exception@std@@6B@", ptr @"??_Gexception@std@@UEAAPEAXI@Z", ptr @"?what@exception@std@@UEBAPEBDXZ"] }, comdat($"??_7exception@std@@6B@")
@"??_R4exception@std@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVexception@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3exception@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R4exception@std@@6B@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_C@_0BC@EOODALEL@Unknown?5exception?$AA@" = linkonce_odr dso_local unnamed_addr constant [18 x i8] c"Unknown exception\00", comdat, align 1
@4 = private unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr @"??_R4failure@ios_base@std@@6B@", ptr @"??_Gfailure@ios_base@std@@UEAAPEAXI@Z", ptr @"?what@exception@std@@UEBAPEBDXZ"] }, comdat($"??_7failure@ios_base@std@@6B@")
@"??_R4failure@ios_base@std@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVfailure@ios_base@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3failure@ios_base@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R4failure@ios_base@std@@6B@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R3failure@ios_base@std@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 0, i32 5, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R2failure@ios_base@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R2failure@ios_base@std@@8" = linkonce_odr constant [6 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@failure@ios_base@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@system_error@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@_System_error@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@runtime_error@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@exception@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0], comdat
@"??_R1A@?0A@EA@failure@ios_base@std@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVfailure@ios_base@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 4, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3failure@ios_base@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R1A@?0A@EA@system_error@std@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVsystem_error@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 3, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3system_error@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R3system_error@std@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 0, i32 4, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R2system_error@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R2system_error@std@@8" = linkonce_odr constant [5 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@system_error@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@_System_error@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@runtime_error@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@exception@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0], comdat
@"??_R1A@?0A@EA@_System_error@std@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AV_System_error@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 2, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3_System_error@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R3_System_error@std@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 0, i32 3, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R2_System_error@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R2_System_error@std@@8" = linkonce_odr constant [4 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@_System_error@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@runtime_error@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@exception@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0], comdat
@"??_R1A@?0A@EA@runtime_error@std@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVruntime_error@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 1, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3runtime_error@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R3runtime_error@std@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 0, i32 2, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R2runtime_error@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R2runtime_error@std@@8" = linkonce_odr constant [3 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@runtime_error@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@exception@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0], comdat
@5 = private unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr @"??_R4system_error@std@@6B@", ptr @"??_Gsystem_error@std@@UEAAPEAXI@Z", ptr @"?what@exception@std@@UEBAPEBDXZ"] }, comdat($"??_7system_error@std@@6B@")
@"??_R4system_error@std@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVsystem_error@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3system_error@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R4system_error@std@@6B@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@6 = private unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr @"??_R4_System_error@std@@6B@", ptr @"??_G_System_error@std@@UEAAPEAXI@Z", ptr @"?what@exception@std@@UEBAPEBDXZ"] }, comdat($"??_7_System_error@std@@6B@")
@"??_R4_System_error@std@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AV_System_error@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3_System_error@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R4_System_error@std@@6B@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_C@_02LMMGGCAJ@?3?5?$AA@" = linkonce_odr dso_local unnamed_addr constant [3 x i8] c": \00", comdat, align 1
@7 = private unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr @"??_R4runtime_error@std@@6B@", ptr @"??_Gruntime_error@std@@UEAAPEAXI@Z", ptr @"?what@exception@std@@UEBAPEBDXZ"] }, comdat($"??_7runtime_error@std@@6B@")
@"??_R4runtime_error@std@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVruntime_error@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3runtime_error@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R4runtime_error@std@@6B@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"?_Psave@?$_Facetptr@V?$ctype@D@std@@@std@@2PEBVfacet@locale@2@EB" = linkonce_odr dso_local local_unnamed_addr global ptr null, comdat, align 8
@"?id@?$ctype@D@std@@2V0locale@2@A" = external dso_local local_unnamed_addr global %"class.std::locale::id", align 8
@"?_Id_cnt@id@locale@std@@0HA" = external dso_local local_unnamed_addr global i32, align 4
@"??_C@_00CNPNBAHC@?$AA@" = linkonce_odr dso_local unnamed_addr constant [1 x i8] zeroinitializer, comdat, align 1
@"??_C@_0BA@ELKIONDK@bad?5locale?5name?$AA@" = linkonce_odr dso_local unnamed_addr constant [16 x i8] c"bad locale name\00", comdat, align 1
@8 = private unnamed_addr constant { [12 x ptr] } { [12 x ptr] [ptr @"??_R4?$ctype@D@std@@6B@", ptr @"??_G?$ctype@D@std@@MEAAPEAXI@Z", ptr @"?_Incref@facet@locale@std@@UEAAXXZ", ptr @"?_Decref@facet@locale@std@@UEAAPEAV_Facet_base@3@XZ", ptr @"?do_tolower@?$ctype@D@std@@MEBAPEBDPEADPEBD@Z", ptr @"?do_tolower@?$ctype@D@std@@MEBADD@Z", ptr @"?do_toupper@?$ctype@D@std@@MEBAPEBDPEADPEBD@Z", ptr @"?do_toupper@?$ctype@D@std@@MEBADD@Z", ptr @"?do_widen@?$ctype@D@std@@MEBAPEBDPEBD0PEAD@Z", ptr @"?do_widen@?$ctype@D@std@@MEBADD@Z", ptr @"?do_narrow@?$ctype@D@std@@MEBAPEBDPEBD0DPEAD@Z", ptr @"?do_narrow@?$ctype@D@std@@MEBADDD@Z"] }, comdat($"??_7?$ctype@D@std@@6B@")
@"??_R4?$ctype@D@std@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AV?$ctype@D@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3?$ctype@D@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R4?$ctype@D@std@@6B@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R0?AV?$ctype@D@std@@@8" = linkonce_odr global %rtti.TypeDescriptor19 { ptr @"??_7type_info@@6B@", ptr null, [20 x i8] c".?AV?$ctype@D@std@@\00" }, comdat
@"??_R3?$ctype@D@std@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 1, i32 5, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R2?$ctype@D@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R2?$ctype@D@std@@8" = linkonce_odr constant [6 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@?$ctype@D@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@ctype_base@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@facet@locale@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@_Facet_base@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R17?0A@EA@_Crt_new_delete@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0], comdat
@"??_R1A@?0A@EA@?$ctype@D@std@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AV?$ctype@D@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 4, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3?$ctype@D@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R1A@?0A@EA@ctype_base@std@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AUctype_base@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 3, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3ctype_base@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R0?AUctype_base@std@@@8" = linkonce_odr global %rtti.TypeDescriptor20 { ptr @"??_7type_info@@6B@", ptr null, [21 x i8] c".?AUctype_base@std@@\00" }, comdat
@"??_R3ctype_base@std@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 1, i32 4, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R2ctype_base@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R2ctype_base@std@@8" = linkonce_odr constant [5 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@ctype_base@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@facet@locale@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@_Facet_base@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R17?0A@EA@_Crt_new_delete@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0], comdat
@"??_R1A@?0A@EA@facet@locale@std@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVfacet@locale@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 2, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3facet@locale@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R0?AVfacet@locale@std@@@8" = linkonce_odr global %rtti.TypeDescriptor22 { ptr @"??_7type_info@@6B@", ptr null, [23 x i8] c".?AVfacet@locale@std@@\00" }, comdat
@"??_R3facet@locale@std@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 1, i32 3, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R2facet@locale@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R2facet@locale@std@@8" = linkonce_odr constant [4 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@facet@locale@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@_Facet_base@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R17?0A@EA@_Crt_new_delete@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0], comdat
@"??_R1A@?0A@EA@_Facet_base@std@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AV_Facet_base@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3_Facet_base@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R0?AV_Facet_base@std@@@8" = linkonce_odr global %rtti.TypeDescriptor21 { ptr @"??_7type_info@@6B@", ptr null, [22 x i8] c".?AV_Facet_base@std@@\00" }, comdat
@"??_R3_Facet_base@std@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 0, i32 1, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R2_Facet_base@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R2_Facet_base@std@@8" = linkonce_odr constant [2 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@_Facet_base@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0], comdat
@"??_R17?0A@EA@_Crt_new_delete@std@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AU_Crt_new_delete@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 8, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3_Crt_new_delete@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R0?AU_Crt_new_delete@std@@@8" = linkonce_odr global %rtti.TypeDescriptor25 { ptr @"??_7type_info@@6B@", ptr null, [26 x i8] c".?AU_Crt_new_delete@std@@\00" }, comdat
@"??_R3_Crt_new_delete@std@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 0, i32 1, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R2_Crt_new_delete@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R2_Crt_new_delete@std@@8" = linkonce_odr constant [2 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@_Crt_new_delete@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0], comdat
@"??_R1A@?0A@EA@_Crt_new_delete@std@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AU_Crt_new_delete@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3_Crt_new_delete@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@9 = private unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr @"??_R4ctype_base@std@@6B@", ptr @"??_Gctype_base@std@@UEAAPEAXI@Z", ptr @"?_Incref@facet@locale@std@@UEAAXXZ", ptr @"?_Decref@facet@locale@std@@UEAAPEAV_Facet_base@3@XZ"] }, comdat($"??_7ctype_base@std@@6B@")
@"??_R4ctype_base@std@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AUctype_base@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3ctype_base@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R4ctype_base@std@@6B@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@10 = private unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr @"??_R4facet@locale@std@@6B@", ptr @"??_Gfacet@locale@std@@MEAAPEAXI@Z", ptr @"?_Incref@facet@locale@std@@UEAAXXZ", ptr @"?_Decref@facet@locale@std@@UEAAPEAV_Facet_base@3@XZ"] }, comdat($"??_7facet@locale@std@@6B@")
@"??_R4facet@locale@std@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVfacet@locale@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3facet@locale@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R4facet@locale@std@@6B@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@11 = private unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr @"??_R4_Facet_base@std@@6B@", ptr @"??_G_Facet_base@std@@UEAAPEAXI@Z", ptr @_purecall, ptr @_purecall] }, comdat($"??_7_Facet_base@std@@6B@")
@"??_R4_Facet_base@std@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AV_Facet_base@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3_Facet_base@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R4_Facet_base@std@@6B@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R0?AVbad_cast@std@@@8" = linkonce_odr global %rtti.TypeDescriptor18 { ptr @"??_7type_info@@6B@", ptr null, [19 x i8] c".?AVbad_cast@std@@\00" }, comdat
@"_CT??_R0?AVbad_cast@std@@@8??0bad_cast@std@@QEAA@AEBV01@@Z24" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVbad_cast@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 -1, i32 0, i32 24, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??0bad_cast@std@@QEAA@AEBV01@@Z" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, section ".xdata", comdat
@"_CTA2?AVbad_cast@std@@" = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.2 { i32 2, [2 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"_CT??_R0?AVbad_cast@std@@@8??0bad_cast@std@@QEAA@AEBV01@@Z24" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"_CT??_R0?AVexception@std@@@8??0exception@std@@QEAA@AEBV01@@Z24" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32)] }, section ".xdata", comdat
@"_TI2?AVbad_cast@std@@" = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??1exception@std@@UEAA@XZ" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"_CTA2?AVbad_cast@std@@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, section ".xdata", comdat
@"??_C@_08EPJLHIJG@bad?5cast?$AA@" = linkonce_odr dso_local unnamed_addr constant [9 x i8] c"bad cast\00", comdat, align 1
@12 = private unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr @"??_R4bad_cast@std@@6B@", ptr @"??_Gbad_cast@std@@UEAAPEAXI@Z", ptr @"?what@exception@std@@UEBAPEBDXZ"] }, comdat($"??_7bad_cast@std@@6B@")
@"??_R4bad_cast@std@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVbad_cast@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3bad_cast@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R4bad_cast@std@@6B@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R3bad_cast@std@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 0, i32 2, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R2bad_cast@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R2bad_cast@std@@8" = linkonce_odr constant [3 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@bad_cast@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@exception@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0], comdat
@"??_R1A@?0A@EA@bad_cast@std@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVbad_cast@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 1, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3bad_cast@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"?_Psave@?$_Facetptr@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@2PEBVfacet@locale@2@EB" = linkonce_odr dso_local local_unnamed_addr global ptr null, comdat, align 8
@13 = private unnamed_addr constant { [12 x ptr] } { [12 x ptr] [ptr @"??_R4?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@6B@", ptr @"??_G?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAAPEAXI@Z", ptr @"?_Incref@facet@locale@std@@UEAAXXZ", ptr @"?_Decref@facet@locale@std@@UEAAPEAV_Facet_base@3@XZ", ptr @"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBX@Z", ptr @"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DO@Z", ptr @"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DN@Z", ptr @"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_K@Z", ptr @"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_J@Z", ptr @"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DK@Z", ptr @"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DJ@Z", ptr @"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_N@Z"] }, comdat($"??_7?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@6B@")
@"??_R4?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R4?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@6B@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R0?AV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@8" = linkonce_odr global %rtti.TypeDescriptor73 { ptr @"??_7type_info@@6B@", ptr null, [74 x i8] c".?AV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@\00" }, comdat
@"??_R3?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 1, i32 4, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R2?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R2?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8" = linkonce_odr constant [5 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@facet@locale@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@_Facet_base@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R17?0A@EA@_Crt_new_delete@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0], comdat
@"??_R1A@?0A@EA@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 3, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_C@_02BBAHNLBA@?$CFp?$AA@" = linkonce_odr dso_local unnamed_addr constant [3 x i8] c"%p\00", comdat, align 1
@"?_Psave@?$_Facetptr@V?$numpunct@D@std@@@std@@2PEBVfacet@locale@2@EB" = linkonce_odr dso_local local_unnamed_addr global ptr null, comdat, align 8
@14 = private unnamed_addr constant { [9 x ptr] } { [9 x ptr] [ptr @"??_R4?$numpunct@D@std@@6B@", ptr @"??_G?$numpunct@D@std@@MEAAPEAXI@Z", ptr @"?_Incref@facet@locale@std@@UEAAXXZ", ptr @"?_Decref@facet@locale@std@@UEAAPEAV_Facet_base@3@XZ", ptr @"?do_decimal_point@?$numpunct@D@std@@MEBADXZ", ptr @"?do_thousands_sep@?$numpunct@D@std@@MEBADXZ", ptr @"?do_grouping@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ", ptr @"?do_falsename@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ", ptr @"?do_truename@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"] }, comdat($"??_7?$numpunct@D@std@@6B@")
@"??_R4?$numpunct@D@std@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AV?$numpunct@D@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3?$numpunct@D@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R4?$numpunct@D@std@@6B@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R0?AV?$numpunct@D@std@@@8" = linkonce_odr global %rtti.TypeDescriptor22 { ptr @"??_7type_info@@6B@", ptr null, [23 x i8] c".?AV?$numpunct@D@std@@\00" }, comdat
@"??_R3?$numpunct@D@std@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 1, i32 4, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R2?$numpunct@D@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R2?$numpunct@D@std@@8" = linkonce_odr constant [5 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@?$numpunct@D@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@facet@locale@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@_Facet_base@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R17?0A@EA@_Crt_new_delete@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0], comdat
@"??_R1A@?0A@EA@?$numpunct@D@std@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AV?$numpunct@D@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 3, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3?$numpunct@D@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_C@_05LAPONLG@false?$AA@" = linkonce_odr dso_local unnamed_addr constant [6 x i8] c"false\00", comdat, align 1
@"??_C@_04LOAJBDKD@true?$AA@" = linkonce_odr dso_local unnamed_addr constant [5 x i8] c"true\00", comdat, align 1
@"??_C@_0BI@CFPLBAOH@invalid?5string?5position?$AA@" = linkonce_odr dso_local unnamed_addr constant [24 x i8] c"invalid string position\00", comdat, align 1
@"?_OptionsStorage@?1??__local_stdio_printf_options@@9@4_KA" = linkonce_odr dso_local global i64 0, comdat, align 8
@"??_C@_02MDKMJEGG@eE?$AA@" = linkonce_odr dso_local unnamed_addr constant [3 x i8] c"eE\00", comdat, align 1
@"??_C@_02OOPEBDOJ@pP?$AA@" = linkonce_odr dso_local unnamed_addr constant [3 x i8] c"pP\00", comdat, align 1
@llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @"??__E?id@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A@@YAXXZ", ptr @"?id@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A" }, { i32, ptr, ptr } { i32 65535, ptr @"??__E?id@?$numpunct@D@std@@2V0locale@2@A@@YAXXZ", ptr @"?id@?$numpunct@D@std@@2V0locale@2@A" }]
@llvm.used = appending global [2 x ptr] [ptr @"?id@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A", ptr @"?id@?$numpunct@D@std@@2V0locale@2@A"], section "llvm.metadata"

@"??_7_Iostream_error_category2@std@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [7 x ptr] }, ptr @0, i32 0, i32 0, i32 1)
@"??_7bad_array_new_length@std@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [3 x ptr] }, ptr @1, i32 0, i32 0, i32 1)
@"??_7bad_alloc@std@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [3 x ptr] }, ptr @2, i32 0, i32 0, i32 1)
@"??_7exception@std@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [3 x ptr] }, ptr @3, i32 0, i32 0, i32 1)
@"??_7failure@ios_base@std@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [3 x ptr] }, ptr @4, i32 0, i32 0, i32 1)
@"??_7system_error@std@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [3 x ptr] }, ptr @5, i32 0, i32 0, i32 1)
@"??_7_System_error@std@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [3 x ptr] }, ptr @6, i32 0, i32 0, i32 1)
@"??_7runtime_error@std@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [3 x ptr] }, ptr @7, i32 0, i32 0, i32 1)
@"??_7?$ctype@D@std@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [12 x ptr] }, ptr @8, i32 0, i32 0, i32 1)
@"??_7ctype_base@std@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [4 x ptr] }, ptr @9, i32 0, i32 0, i32 1)
@"??_7facet@locale@std@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [4 x ptr] }, ptr @10, i32 0, i32 0, i32 1)
@"??_7_Facet_base@std@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [4 x ptr] }, ptr @11, i32 0, i32 0, i32 1)
@"??_7bad_cast@std@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [3 x ptr] }, ptr @12, i32 0, i32 0, i32 1)
@"??_7?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [12 x ptr] }, ptr @13, i32 0, i32 0, i32 1)
@"??_7?$numpunct@D@std@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [9 x ptr] }, ptr @14, i32 0, i32 0, i32 1)

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = alloca i32, align 4
  %2 = alloca [5 x i32], align 16
  %3 = alloca [3 x ptr], align 16
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %1) #10
  store i32 42, ptr %1, align 4
  %4 = tail call noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(8) @"?cout@std@@3V?$basic_ostream@DU?$char_traits@D@std@@@1@A", ptr noundef nonnull @"??_C@_0BN@GIAMKINO@1?4?5Basic?5pointer?5operations?3?$AA@")
  %5 = tail call noundef nonnull align 8 dereferenceable(8) ptr @"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %4)
  %6 = tail call noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(8) @"?cout@std@@3V?$basic_ostream@DU?$char_traits@D@std@@@1@A", ptr noundef nonnull @"??_C@_0P@PPPILKBG@Value?5of?5num?3?5?$AA@")
  %7 = load i32, ptr %1, align 4
  %8 = tail call noundef nonnull align 8 dereferenceable(8) ptr @"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z"(ptr noundef nonnull align 8 dereferenceable(8) %6, i32 noundef %7)
  %9 = tail call noundef nonnull align 8 dereferenceable(8) ptr @"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %8)
  %10 = tail call noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(8) @"?cout@std@@3V?$basic_ostream@DU?$char_traits@D@std@@@1@A", ptr noundef nonnull @"??_C@_0BB@IGENIPIM@Address?5of?5num?3?5?$AA@")
  %11 = call noundef nonnull align 8 dereferenceable(8) ptr @"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@PEBX@Z"(ptr noundef nonnull align 8 dereferenceable(8) %10, ptr noundef nonnull %1)
  %12 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %11)
  %13 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(8) @"?cout@std@@3V?$basic_ostream@DU?$char_traits@D@std@@@1@A", ptr noundef nonnull @"??_C@_0P@PDDOLHIE@Value?5of?5ptr?3?5?$AA@")
  %14 = call noundef nonnull align 8 dereferenceable(8) ptr @"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@PEBX@Z"(ptr noundef nonnull align 8 dereferenceable(8) %13, ptr noundef nonnull %1)
  %15 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %14)
  %16 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(8) @"?cout@std@@3V?$basic_ostream@DU?$char_traits@D@std@@@1@A", ptr noundef nonnull @"??_C@_0BH@DIAOJILP@Value?5pointed?5by?5ptr?3?5?$AA@")
  %17 = load i32, ptr %1, align 4
  %18 = call noundef nonnull align 8 dereferenceable(8) ptr @"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z"(ptr noundef nonnull align 8 dereferenceable(8) %16, i32 noundef %17)
  %19 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %18)
  %20 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %19)
  store i32 100, ptr %1, align 4
  %21 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(8) @"?cout@std@@3V?$basic_ostream@DU?$char_traits@D@std@@@1@A", ptr noundef nonnull @"??_C@_0CJ@OJKBMIMM@2?4?5After?5dereferencing?5and?5modif@")
  %22 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %21)
  %23 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(8) @"?cout@std@@3V?$basic_ostream@DU?$char_traits@D@std@@@1@A", ptr noundef nonnull @"??_C@_0BD@PDLLNEEM@New?5value?5of?5num?3?5?$AA@")
  %24 = load i32, ptr %1, align 4
  %25 = call noundef nonnull align 8 dereferenceable(8) ptr @"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z"(ptr noundef nonnull align 8 dereferenceable(8) %23, i32 noundef %24)
  %26 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %25)
  %27 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %26)
  call void @llvm.lifetime.start.p0(i64 20, ptr nonnull %2) #10
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 16 dereferenceable(20) %2, ptr noundef nonnull align 16 dereferenceable(20) @__const.main.arr, i64 20, i1 false)
  %28 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(8) @"?cout@std@@3V?$basic_ostream@DU?$char_traits@D@std@@@1@A", ptr noundef nonnull @"??_C@_0CC@HLHDMDK@3?4?5Pointer?5arithmetic?5with?5array@")
  %29 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %28)
  %30 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(8) @"?cout@std@@3V?$basic_ostream@DU?$char_traits@D@std@@@1@A", ptr noundef nonnull @"??_C@_0BA@COOFDBFC@First?5element?3?5?$AA@")
  %31 = load i32, ptr %2, align 16
  %32 = call noundef nonnull align 8 dereferenceable(8) ptr @"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z"(ptr noundef nonnull align 8 dereferenceable(8) %30, i32 noundef %31)
  %33 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(8) %32, ptr noundef nonnull @"??_C@_04LNPKFDKO@?5at?5?$AA@")
  %34 = call noundef nonnull align 8 dereferenceable(8) ptr @"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@PEBX@Z"(ptr noundef nonnull align 8 dereferenceable(8) %33, ptr noundef nonnull %2)
  %35 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %34)
  %36 = getelementptr inbounds nuw i8, ptr %2, i64 4
  %37 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(8) @"?cout@std@@3V?$basic_ostream@DU?$char_traits@D@std@@@1@A", ptr noundef nonnull @"??_C@_0BB@GFLPPJEB@Second?5element?3?5?$AA@")
  %38 = load i32, ptr %36, align 4
  %39 = call noundef nonnull align 8 dereferenceable(8) ptr @"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z"(ptr noundef nonnull align 8 dereferenceable(8) %37, i32 noundef %38)
  %40 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(8) %39, ptr noundef nonnull @"??_C@_04LNPKFDKO@?5at?5?$AA@")
  %41 = call noundef nonnull align 8 dereferenceable(8) ptr @"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@PEBX@Z"(ptr noundef nonnull align 8 dereferenceable(8) %40, ptr noundef nonnull %36)
  %42 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %41)
  %43 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(8) @"?cout@std@@3V?$basic_ostream@DU?$char_traits@D@std@@@1@A", ptr noundef nonnull @"??_C@_0BA@OJNMCLP@Third?5element?3?5?$AA@")
  %44 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %45 = load i32, ptr %44, align 8
  %46 = call noundef nonnull align 8 dereferenceable(8) ptr @"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z"(ptr noundef nonnull align 8 dereferenceable(8) %43, i32 noundef %45)
  %47 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(8) %46, ptr noundef nonnull @"??_C@_04LNPKFDKO@?5at?5?$AA@")
  %48 = call noundef nonnull align 8 dereferenceable(8) ptr @"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@PEBX@Z"(ptr noundef nonnull align 8 dereferenceable(8) %47, ptr noundef nonnull %44)
  %49 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %48)
  %50 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %49)
  %51 = call noalias noundef nonnull dereferenceable(4) ptr @"??2@YAPEAX_K@Z"(i64 noundef 4) #27
  store i32 200, ptr %51, align 4
  %52 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(8) @"?cout@std@@3V?$basic_ostream@DU?$char_traits@D@std@@@1@A", ptr noundef nonnull @"??_C@_0BO@LFEDEBBC@4?4?5Dynamic?5memory?5allocation?3?$AA@")
  %53 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %52)
  %54 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(8) @"?cout@std@@3V?$basic_ostream@DU?$char_traits@D@std@@@1@A", ptr noundef nonnull @"??_C@_0BO@NBEBKJHP@Dynamically?5allocated?5value?3?5?$AA@")
  %55 = load i32, ptr %51, align 4
  %56 = call noundef nonnull align 8 dereferenceable(8) ptr @"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z"(ptr noundef nonnull align 8 dereferenceable(8) %54, i32 noundef %55)
  %57 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(8) %56, ptr noundef nonnull @"??_C@_04LNPKFDKO@?5at?5?$AA@")
  %58 = call noundef nonnull align 8 dereferenceable(8) ptr @"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@PEBX@Z"(ptr noundef nonnull align 8 dereferenceable(8) %57, ptr noundef nonnull %51)
  %59 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %58)
  %60 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(8) @"?cout@std@@3V?$basic_ostream@DU?$char_traits@D@std@@@1@A", ptr noundef nonnull @"??_C@_0BH@LDGOONGC@5?4?5Pointers?5and?5const?3?$AA@")
  %61 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %60)
  %62 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(8) @"?cout@std@@3V?$basic_ostream@DU?$char_traits@D@std@@@1@A", ptr noundef nonnull @"??_C@_0CB@GFNKDNIK@Constant?5value?5through?5pointer?3?5@")
  %63 = call noundef nonnull align 8 dereferenceable(8) ptr @"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z"(ptr noundef nonnull align 8 dereferenceable(8) %62, i32 noundef 300)
  %64 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %63)
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %3) #10
  store ptr %1, ptr %3, align 16
  %65 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store ptr %36, ptr %65, align 8
  %66 = getelementptr inbounds nuw i8, ptr %3, i64 16
  store ptr %51, ptr %66, align 16
  %67 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(8) @"?cout@std@@3V?$basic_ostream@DU?$char_traits@D@std@@@1@A", ptr noundef nonnull @"??_C@_0BG@OMIOAIGA@6?4?5Array?5of?5pointers?3?$AA@")
  %68 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %67)
  br label %78

69:                                               ; preds = %78
  %70 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z"(ptr noundef nonnull align 8 dereferenceable(8) @"?cout@std@@3V?$basic_ostream@DU?$char_traits@D@std@@@1@A")
  %71 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(8) @"?cout@std@@3V?$basic_ostream@DU?$char_traits@D@std@@@1@A", ptr noundef nonnull @"??_C@_0BH@IEOBKIOK@7?4?5Pointer?5comparison?3?$AA@")
  %72 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %71)
  %73 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(8) @"?cout@std@@3V?$basic_ostream@DU?$char_traits@D@std@@@1@A", ptr noundef nonnull @"??_C@_0BH@DGOHBFGE@ptr?5is?5pointing?5to?5num?$AA@")
  %74 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %73)
  %75 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(8) @"?cout@std@@3V?$basic_ostream@DU?$char_traits@D@std@@@1@A", ptr noundef nonnull @"??_C@_0CB@ILBHDHPI@arrPtr?5is?5ahead?5of?5arr?5in?5memory@")
  %76 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %75)
  %77 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z"(ptr noundef nonnull align 8 dereferenceable(8) @"?cout@std@@3V?$basic_ostream@DU?$char_traits@D@std@@@1@A")
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %51, i64 noundef 4) #28
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %3) #10
  call void @llvm.lifetime.end.p0(i64 20, ptr nonnull %2) #10
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %1) #10
  ret i32 0

78:                                               ; preds = %0, %78
  %79 = phi i64 [ 0, %0 ], [ %89, %78 ]
  %80 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(8) @"?cout@std@@3V?$basic_ostream@DU?$char_traits@D@std@@@1@A", ptr noundef nonnull @"??_C@_08IMOEKGIO@Pointer?5?$AA@")
  %81 = trunc nuw nsw i64 %79 to i32
  %82 = call noundef nonnull align 8 dereferenceable(8) ptr @"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z"(ptr noundef nonnull align 8 dereferenceable(8) %80, i32 noundef %81)
  %83 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(8) %82, ptr noundef nonnull @"??_C@_0BD@INFLFJBG@?5points?5to?5value?3?5?$AA@")
  %84 = getelementptr inbounds nuw [3 x ptr], ptr %3, i64 0, i64 %79
  %85 = load ptr, ptr %84, align 8
  %86 = load i32, ptr %85, align 4
  %87 = call noundef nonnull align 8 dereferenceable(8) ptr @"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z"(ptr noundef nonnull align 8 dereferenceable(8) %83, i32 noundef %86)
  %88 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %87)
  %89 = add nuw nsw i64 %79, 1
  %90 = icmp eq i64 %89, 3
  br i1 %90, label %69, label %78, !llvm.loop !15
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #1

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0, ptr noundef %1) local_unnamed_addr #2 comdat personality ptr @__CxxFrameHandler3 {
  %3 = alloca %"class.std::error_code", align 8
  %4 = alloca %"class.std::ios_base::failure", align 8
  %5 = alloca %"class.std::error_code", align 8
  %6 = alloca %"class.std::basic_ostream<char>::sentry", align 8
  %7 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %1) #10
  %8 = load ptr, ptr %0, align 8
  %9 = getelementptr inbounds nuw i8, ptr %8, i64 4
  %10 = load i32, ptr %9, align 4
  %11 = sext i32 %10 to i64
  %12 = getelementptr inbounds i8, ptr %0, i64 %11
  %13 = getelementptr inbounds nuw i8, ptr %12, i64 40
  %14 = load i64, ptr %13, align 8
  %15 = icmp sgt i64 %14, 0
  %16 = icmp sgt i64 %14, %7
  %17 = and i1 %15, %16
  %18 = sub nsw i64 %14, %7
  %19 = select i1 %17, i64 %18, i64 0
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %6) #10
  store ptr %0, ptr %6, align 8
  %20 = load i32, ptr %9, align 4
  %21 = sext i32 %20 to i64
  %22 = getelementptr inbounds i8, ptr %0, i64 %21
  %23 = getelementptr inbounds nuw i8, ptr %22, i64 72
  %24 = load ptr, ptr %23, align 8
  %25 = icmp eq ptr %24, null
  br i1 %25, label %30, label %26

26:                                               ; preds = %2
  %27 = load ptr, ptr %24, align 8
  %28 = getelementptr inbounds nuw i8, ptr %27, i64 8
  %29 = load ptr, ptr %28, align 8
  tail call void %29(ptr noundef nonnull align 8 dereferenceable(104) %24)
  br label %30

30:                                               ; preds = %26, %2
  %31 = load ptr, ptr %0, align 8
  %32 = getelementptr inbounds nuw i8, ptr %31, i64 4
  %33 = load i32, ptr %32, align 4
  %34 = sext i32 %33 to i64
  %35 = getelementptr inbounds i8, ptr %0, i64 %34
  %36 = getelementptr inbounds nuw i8, ptr %35, i64 16
  %37 = load i32, ptr %36, align 8
  %38 = icmp eq i32 %37, 0
  br i1 %38, label %41, label %39

39:                                               ; preds = %30
  %40 = getelementptr inbounds nuw i8, ptr %6, i64 8
  store i8 0, ptr %40, align 8
  br label %64

41:                                               ; preds = %30
  %42 = getelementptr inbounds nuw i8, ptr %35, i64 80
  %43 = load ptr, ptr %42, align 8
  %44 = icmp eq ptr %43, null
  %45 = icmp eq ptr %43, %0
  %46 = or i1 %44, %45
  br i1 %46, label %47, label %49

47:                                               ; preds = %41
  %48 = getelementptr inbounds nuw i8, ptr %6, i64 8
  store i8 1, ptr %48, align 8
  br label %64

49:                                               ; preds = %41
  %50 = invoke noundef nonnull align 8 dereferenceable(8) ptr @"?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %43)
          to label %51 unwind label %62

51:                                               ; preds = %49
  %52 = load ptr, ptr %0, align 8
  %53 = getelementptr inbounds nuw i8, ptr %52, i64 4
  %54 = load i32, ptr %53, align 4
  %55 = sext i32 %54 to i64
  %56 = getelementptr inbounds i8, ptr %0, i64 %55
  %57 = getelementptr inbounds nuw i8, ptr %56, i64 16
  %58 = load i32, ptr %57, align 8
  %59 = icmp eq i32 %58, 0
  %60 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %61 = zext i1 %59 to i8
  store i8 %61, ptr %60, align 8
  br label %64

62:                                               ; preds = %49
  %63 = cleanuppad within none []
  call void @"??1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %6) #10 [ "funclet"(token %63) ]
  cleanupret from %63 unwind to caller

64:                                               ; preds = %39, %47, %51
  %65 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %66 = load i8, ptr %65, align 8, !range !18, !noundef !19
  %67 = trunc nuw i8 %66 to i1
  br i1 %67, label %68, label %199

68:                                               ; preds = %64
  %69 = load ptr, ptr %0, align 8
  %70 = getelementptr inbounds nuw i8, ptr %69, i64 4
  %71 = load i32, ptr %70, align 4
  %72 = sext i32 %71 to i64
  %73 = getelementptr inbounds i8, ptr %0, i64 %72
  %74 = getelementptr inbounds nuw i8, ptr %73, i64 24
  %75 = load i32, ptr %74, align 8
  %76 = and i32 %75, 448
  %77 = icmp ne i32 %76, 64
  %78 = icmp sgt i64 %19, 0
  %79 = select i1 %77, i1 %78, i1 false
  br i1 %79, label %80, label %118

80:                                               ; preds = %68, %115
  %81 = phi i64 [ %116, %115 ], [ %19, %68 ]
  %82 = load ptr, ptr %0, align 8
  %83 = getelementptr inbounds nuw i8, ptr %82, i64 4
  %84 = load i32, ptr %83, align 4
  %85 = sext i32 %84 to i64
  %86 = getelementptr inbounds i8, ptr %0, i64 %85
  %87 = getelementptr inbounds nuw i8, ptr %86, i64 72
  %88 = load ptr, ptr %87, align 8
  %89 = getelementptr inbounds nuw i8, ptr %86, i64 88
  %90 = load i8, ptr %89, align 8
  %91 = getelementptr inbounds nuw i8, ptr %88, i64 64
  %92 = load ptr, ptr %91, align 8
  %93 = load ptr, ptr %92, align 8
  %94 = icmp eq ptr %93, null
  br i1 %94, label %106, label %95

95:                                               ; preds = %80
  %96 = getelementptr inbounds nuw i8, ptr %88, i64 88
  %97 = load ptr, ptr %96, align 8
  %98 = load i32, ptr %97, align 4
  %99 = icmp sgt i32 %98, 0
  br i1 %99, label %100, label %106

100:                                              ; preds = %95
  %101 = add nsw i32 %98, -1
  store i32 %101, ptr %97, align 4
  %102 = load ptr, ptr %91, align 8
  %103 = load ptr, ptr %102, align 8
  %104 = getelementptr inbounds nuw i8, ptr %103, i64 1
  store ptr %104, ptr %102, align 8
  store i8 %90, ptr %103, align 1
  %105 = zext i8 %90 to i32
  br label %112

106:                                              ; preds = %95, %80
  %107 = zext i8 %90 to i32
  %108 = load ptr, ptr %88, align 8
  %109 = getelementptr inbounds nuw i8, ptr %108, i64 24
  %110 = load ptr, ptr %109, align 8
  %111 = invoke noundef i32 %110(ptr noundef nonnull align 8 dereferenceable(104) %88, i32 noundef %107)
          to label %112 unwind label %174

112:                                              ; preds = %100, %106
  %113 = phi i32 [ %105, %100 ], [ %111, %106 ]
  %114 = icmp ne i32 %113, -1
  br i1 %114, label %115, label %118

115:                                              ; preds = %112
  %116 = add nsw i64 %81, -1
  %117 = icmp sgt i64 %81, 1
  br i1 %117, label %80, label %118

118:                                              ; preds = %112, %115, %68
  %119 = phi i1 [ true, %68 ], [ %114, %115 ], [ %114, %112 ]
  %120 = phi i32 [ 0, %68 ], [ 4, %112 ], [ 0, %115 ]
  %121 = phi i64 [ %19, %68 ], [ %81, %112 ], [ 0, %115 ]
  br i1 %119, label %122, label %137

122:                                              ; preds = %118
  %123 = load ptr, ptr %0, align 8
  %124 = getelementptr inbounds nuw i8, ptr %123, i64 4
  %125 = load i32, ptr %124, align 4
  %126 = sext i32 %125 to i64
  %127 = getelementptr inbounds i8, ptr %0, i64 %126
  %128 = getelementptr inbounds nuw i8, ptr %127, i64 72
  %129 = load ptr, ptr %128, align 8
  %130 = load ptr, ptr %129, align 8
  %131 = getelementptr inbounds nuw i8, ptr %130, i64 72
  %132 = load ptr, ptr %131, align 8
  %133 = invoke noundef i64 %132(ptr noundef nonnull align 8 dereferenceable(104) %129, ptr noundef nonnull %1, i64 noundef %7)
          to label %134 unwind label %174

134:                                              ; preds = %122
  %135 = icmp eq i64 %133, %7
  %136 = select i1 %135, i32 0, i32 4
  br label %137

137:                                              ; preds = %134, %118
  %138 = phi i32 [ %120, %118 ], [ %136, %134 ]
  %139 = icmp eq i32 %138, 0
  %140 = icmp sgt i64 %121, 0
  %141 = select i1 %139, i1 %140, i1 false
  br i1 %141, label %142, label %191

142:                                              ; preds = %137, %188
  %143 = phi i64 [ %189, %188 ], [ %121, %137 ]
  %144 = load ptr, ptr %0, align 8
  %145 = getelementptr inbounds nuw i8, ptr %144, i64 4
  %146 = load i32, ptr %145, align 4
  %147 = sext i32 %146 to i64
  %148 = getelementptr inbounds i8, ptr %0, i64 %147
  %149 = getelementptr inbounds nuw i8, ptr %148, i64 72
  %150 = load ptr, ptr %149, align 8
  %151 = getelementptr inbounds nuw i8, ptr %148, i64 88
  %152 = load i8, ptr %151, align 8
  %153 = getelementptr inbounds nuw i8, ptr %150, i64 64
  %154 = load ptr, ptr %153, align 8
  %155 = load ptr, ptr %154, align 8
  %156 = icmp eq ptr %155, null
  br i1 %156, label %168, label %157

157:                                              ; preds = %142
  %158 = getelementptr inbounds nuw i8, ptr %150, i64 88
  %159 = load ptr, ptr %158, align 8
  %160 = load i32, ptr %159, align 4
  %161 = icmp sgt i32 %160, 0
  br i1 %161, label %162, label %168

162:                                              ; preds = %157
  %163 = add nsw i32 %160, -1
  store i32 %163, ptr %159, align 4
  %164 = load ptr, ptr %153, align 8
  %165 = load ptr, ptr %164, align 8
  %166 = getelementptr inbounds nuw i8, ptr %165, i64 1
  store ptr %166, ptr %164, align 8
  store i8 %152, ptr %165, align 1
  %167 = zext i8 %152 to i32
  br label %185

168:                                              ; preds = %157, %142
  %169 = zext i8 %152 to i32
  %170 = load ptr, ptr %150, align 8
  %171 = getelementptr inbounds nuw i8, ptr %170, i64 24
  %172 = load ptr, ptr %171, align 8
  %173 = invoke noundef i32 %172(ptr noundef nonnull align 8 dereferenceable(104) %150, i32 noundef %169)
          to label %185 unwind label %174

174:                                              ; preds = %168, %122, %106
  %175 = phi i32 [ 0, %106 ], [ 0, %122 ], [ %138, %168 ]
  %176 = catchswitch within none [label %177] unwind label %250

177:                                              ; preds = %174
  %178 = catchpad within %176 [ptr null, i32 64, ptr null]
  %179 = load ptr, ptr %0, align 8
  %180 = getelementptr inbounds nuw i8, ptr %179, i64 4
  %181 = load i32, ptr %180, align 4
  %182 = sext i32 %181 to i64
  %183 = getelementptr inbounds i8, ptr %0, i64 %182
  invoke void @"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"(ptr noundef nonnull align 8 dereferenceable(96) %183, i32 noundef 4, i1 noundef zeroext true) [ "funclet"(token %178) ]
          to label %184 unwind label %250

184:                                              ; preds = %177
  catchret from %178 to label %199

185:                                              ; preds = %162, %168
  %186 = phi i32 [ %167, %162 ], [ %173, %168 ]
  %187 = icmp eq i32 %186, -1
  br i1 %187, label %191, label %188

188:                                              ; preds = %185
  %189 = add nsw i64 %143, -1
  %190 = icmp sgt i64 %143, 1
  br i1 %190, label %142, label %191

191:                                              ; preds = %185, %188, %137
  %192 = phi i32 [ %138, %137 ], [ %138, %188 ], [ 4, %185 ]
  %193 = load ptr, ptr %0, align 8
  %194 = getelementptr inbounds nuw i8, ptr %193, i64 4
  %195 = load i32, ptr %194, align 4
  %196 = sext i32 %195 to i64
  %197 = getelementptr inbounds i8, ptr %0, i64 %196
  %198 = getelementptr inbounds nuw i8, ptr %197, i64 40
  store i64 0, ptr %198, align 8
  br label %199

199:                                              ; preds = %64, %191, %184
  %200 = phi i32 [ %192, %191 ], [ %175, %184 ], [ 4, %64 ]
  %201 = load ptr, ptr %0, align 8
  %202 = getelementptr inbounds nuw i8, ptr %201, i64 4
  %203 = load i32, ptr %202, align 4
  %204 = sext i32 %203 to i64
  %205 = getelementptr inbounds i8, ptr %0, i64 %204
  %206 = getelementptr inbounds nuw i8, ptr %205, i64 16
  %207 = load i32, ptr %206, align 8
  %208 = or i32 %207, %200
  %209 = getelementptr inbounds nuw i8, ptr %205, i64 72
  %210 = load ptr, ptr %209, align 8
  %211 = icmp eq ptr %210, null
  %212 = select i1 %211, i32 4, i32 0
  call void @llvm.lifetime.start.p0(i64 40, ptr nonnull %4)
  %213 = and i32 %208, 23
  %214 = or i32 %212, %213
  store i32 %214, ptr %206, align 8
  %215 = getelementptr inbounds nuw i8, ptr %205, i64 20
  %216 = load i32, ptr %215, align 4
  %217 = and i32 %216, %214
  %218 = icmp eq i32 %217, 0
  br i1 %218, label %229, label %219

219:                                              ; preds = %199
  %220 = and i32 %217, 4
  %221 = icmp eq i32 %220, 0
  %222 = and i32 %217, 2
  %223 = icmp eq i32 %222, 0
  %224 = select i1 %223, ptr @"??_C@_0BF@OOHOMBOF@ios_base?3?3eofbit?5set?$AA@", ptr @"??_C@_0BG@FMKFHCIL@ios_base?3?3failbit?5set?$AA@"
  %225 = select i1 %221, ptr %224, ptr @"??_C@_0BF@PHHKMMFD@ios_base?3?3badbit?5set?$AA@"
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %5) #10
  call void @"?make_error_code@std@@YA?AVerror_code@1@W4io_errc@1@@Z"(ptr dead_on_unwind nonnull writable sret(%"class.std::error_code") align 8 %5, i32 noundef 1) #10
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %3)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %3, ptr noundef nonnull align 8 dereferenceable(16) %5, i64 16, i1 false)
  %226 = invoke noundef ptr @"??0system_error@std@@QEAA@Verror_code@1@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(40) %4, ptr dead_on_return noundef nonnull %3, ptr noundef nonnull %225)
          to label %227 unwind label %250

227:                                              ; preds = %219
  store ptr @"??_7failure@ios_base@std@@6B@", ptr %4, align 8
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %3)
  invoke void @_CxxThrowException(ptr nonnull %4, ptr nonnull @"_TI5?AVfailure@ios_base@std@@") #29
          to label %228 unwind label %250

228:                                              ; preds = %227
  unreachable

229:                                              ; preds = %199
  call void @llvm.lifetime.end.p0(i64 40, ptr nonnull %4)
  %230 = tail call noundef zeroext i1 @"?uncaught_exception@std@@YA_NXZ"() #10
  br i1 %230, label %233, label %231

231:                                              ; preds = %229
  %232 = load ptr, ptr %6, align 8, !nonnull !19, !align !20
  tail call void @"?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ"(ptr noundef nonnull align 8 dereferenceable(8) %232) #10
  br label %233

233:                                              ; preds = %231, %229
  %234 = load ptr, ptr %6, align 8, !nonnull !19, !align !20
  %235 = load ptr, ptr %234, align 8
  %236 = getelementptr inbounds nuw i8, ptr %235, i64 4
  %237 = load i32, ptr %236, align 4
  %238 = sext i32 %237 to i64
  %239 = getelementptr inbounds i8, ptr %234, i64 %238
  %240 = getelementptr inbounds nuw i8, ptr %239, i64 72
  %241 = load ptr, ptr %240, align 8
  %242 = icmp eq ptr %241, null
  br i1 %242, label %249, label %243

243:                                              ; preds = %233
  %244 = load ptr, ptr %241, align 8
  %245 = getelementptr inbounds nuw i8, ptr %244, i64 16
  %246 = load ptr, ptr %245, align 8
  invoke void %246(ptr noundef nonnull align 8 dereferenceable(104) %241)
          to label %249 unwind label %247

247:                                              ; preds = %243
  %248 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %248) ]
  unreachable

249:                                              ; preds = %233, %243
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %6) #10
  ret ptr %0

250:                                              ; preds = %219, %227, %177, %174
  %251 = cleanuppad within none []
  call void @"??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %6) #10 [ "funclet"(token %251) ]
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %6) #10
  cleanupret from %251 unwind to caller
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(8) ptr @"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0) local_unnamed_addr #2 comdat personality ptr @__CxxFrameHandler3 {
  %2 = alloca %"class.std::locale", align 8
  %3 = load ptr, ptr %0, align 8
  %4 = getelementptr inbounds nuw i8, ptr %3, i64 4
  %5 = load i32, ptr %4, align 4
  %6 = sext i32 %5 to i64
  %7 = getelementptr inbounds i8, ptr %0, i64 %6
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %2) #10
  tail call void @llvm.experimental.noalias.scope.decl(metadata !21)
  %8 = getelementptr inbounds nuw i8, ptr %7, i64 64
  %9 = load ptr, ptr %8, align 8, !noalias !21
  %10 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %11 = getelementptr inbounds nuw i8, ptr %9, i64 8
  %12 = load ptr, ptr %11, align 8, !noalias !21
  store ptr %12, ptr %10, align 8, !alias.scope !21
  %13 = load ptr, ptr %12, align 8, !noalias !21
  %14 = getelementptr inbounds nuw i8, ptr %13, i64 8
  %15 = load ptr, ptr %14, align 8, !noalias !21
  tail call void %15(ptr noundef nonnull align 8 dereferenceable(16) %12) #10, !noalias !21
  %16 = invoke noundef nonnull align 8 dereferenceable(48) ptr @"??$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z"(ptr noundef nonnull align 8 dereferenceable(16) %2)
          to label %17 unwind label %35

17:                                               ; preds = %1
  %18 = load ptr, ptr %16, align 8
  %19 = getelementptr inbounds nuw i8, ptr %18, i64 64
  %20 = load ptr, ptr %19, align 8
  %21 = invoke noundef i8 %20(ptr noundef nonnull align 8 dereferenceable(48) %16, i8 noundef 10)
          to label %22 unwind label %35

22:                                               ; preds = %17
  %23 = load ptr, ptr %10, align 8
  %24 = icmp eq ptr %23, null
  br i1 %24, label %50, label %25

25:                                               ; preds = %22
  %26 = load ptr, ptr %23, align 8
  %27 = getelementptr inbounds nuw i8, ptr %26, i64 16
  %28 = load ptr, ptr %27, align 8
  %29 = call noundef ptr %28(ptr noundef nonnull align 8 dereferenceable(16) %23) #10
  %30 = icmp eq ptr %29, null
  br i1 %30, label %50, label %31

31:                                               ; preds = %25
  %32 = load ptr, ptr %29, align 8
  %33 = load ptr, ptr %32, align 8
  %34 = call noundef ptr %33(ptr noundef nonnull align 8 dereferenceable(8) %29, i32 noundef 1) #10
  br label %50

35:                                               ; preds = %17, %1
  %36 = cleanuppad within none []
  %37 = load ptr, ptr %10, align 8
  %38 = icmp eq ptr %37, null
  br i1 %38, label %49, label %39

39:                                               ; preds = %35
  %40 = load ptr, ptr %37, align 8
  %41 = getelementptr inbounds nuw i8, ptr %40, i64 16
  %42 = load ptr, ptr %41, align 8
  %43 = call noundef ptr %42(ptr noundef nonnull align 8 dereferenceable(16) %37) #10 [ "funclet"(token %36) ]
  %44 = icmp eq ptr %43, null
  br i1 %44, label %49, label %45

45:                                               ; preds = %39
  %46 = load ptr, ptr %43, align 8
  %47 = load ptr, ptr %46, align 8
  %48 = call noundef ptr %47(ptr noundef nonnull align 8 dereferenceable(8) %43, i32 noundef 1) #10 [ "funclet"(token %36) ]
  br label %49

49:                                               ; preds = %45, %39, %35
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %2) #10
  cleanupret from %36 unwind to caller

50:                                               ; preds = %22, %25, %31
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %2) #10
  %51 = call noundef nonnull align 8 dereferenceable(8) ptr @"?put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0, i8 noundef %21)
  %52 = call noundef nonnull align 8 dereferenceable(8) ptr @"?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %0)
  ret ptr %0
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(8) ptr @"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0, i32 noundef %1) local_unnamed_addr #2 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %3 = alloca %"class.std::error_code", align 8
  %4 = alloca %"class.std::ios_base::failure", align 8
  %5 = alloca %"class.std::error_code", align 8
  %6 = alloca %"class.std::ostreambuf_iterator", align 8
  %7 = alloca %"class.std::basic_ostream<char>::sentry", align 8
  %8 = alloca %"class.std::locale", align 8
  %9 = alloca %"class.std::ostreambuf_iterator", align 8
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %7) #10
  store ptr %0, ptr %7, align 8
  %10 = load ptr, ptr %0, align 8
  %11 = getelementptr inbounds nuw i8, ptr %10, i64 4
  %12 = load i32, ptr %11, align 4
  %13 = sext i32 %12 to i64
  %14 = getelementptr inbounds i8, ptr %0, i64 %13
  %15 = getelementptr inbounds nuw i8, ptr %14, i64 72
  %16 = load ptr, ptr %15, align 8
  %17 = icmp eq ptr %16, null
  br i1 %17, label %22, label %18

18:                                               ; preds = %2
  %19 = load ptr, ptr %16, align 8
  %20 = getelementptr inbounds nuw i8, ptr %19, i64 8
  %21 = load ptr, ptr %20, align 8
  tail call void %21(ptr noundef nonnull align 8 dereferenceable(104) %16)
  br label %22

22:                                               ; preds = %18, %2
  %23 = load ptr, ptr %0, align 8
  %24 = getelementptr inbounds nuw i8, ptr %23, i64 4
  %25 = load i32, ptr %24, align 4
  %26 = sext i32 %25 to i64
  %27 = getelementptr inbounds i8, ptr %0, i64 %26
  %28 = getelementptr inbounds nuw i8, ptr %27, i64 16
  %29 = load i32, ptr %28, align 8
  %30 = icmp eq i32 %29, 0
  br i1 %30, label %33, label %31

31:                                               ; preds = %22
  %32 = getelementptr inbounds nuw i8, ptr %7, i64 8
  store i8 0, ptr %32, align 8
  br label %56

33:                                               ; preds = %22
  %34 = getelementptr inbounds nuw i8, ptr %27, i64 80
  %35 = load ptr, ptr %34, align 8
  %36 = icmp eq ptr %35, null
  %37 = icmp eq ptr %35, %0
  %38 = or i1 %36, %37
  br i1 %38, label %39, label %41

39:                                               ; preds = %33
  %40 = getelementptr inbounds nuw i8, ptr %7, i64 8
  store i8 1, ptr %40, align 8
  br label %56

41:                                               ; preds = %33
  %42 = invoke noundef nonnull align 8 dereferenceable(8) ptr @"?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %35)
          to label %43 unwind label %54

43:                                               ; preds = %41
  %44 = load ptr, ptr %0, align 8
  %45 = getelementptr inbounds nuw i8, ptr %44, i64 4
  %46 = load i32, ptr %45, align 4
  %47 = sext i32 %46 to i64
  %48 = getelementptr inbounds i8, ptr %0, i64 %47
  %49 = getelementptr inbounds nuw i8, ptr %48, i64 16
  %50 = load i32, ptr %49, align 8
  %51 = icmp eq i32 %50, 0
  %52 = getelementptr inbounds nuw i8, ptr %7, i64 8
  %53 = zext i1 %51 to i8
  store i8 %53, ptr %52, align 8
  br label %56

54:                                               ; preds = %41
  %55 = cleanuppad within none []
  call void @"??1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %7) #10 [ "funclet"(token %55) ]
  cleanupret from %55 unwind to caller

56:                                               ; preds = %31, %39, %43
  %57 = getelementptr inbounds nuw i8, ptr %7, i64 8
  %58 = load i8, ptr %57, align 8, !range !18, !noundef !19
  %59 = trunc nuw i8 %58 to i1
  br i1 %59, label %60, label %131

60:                                               ; preds = %56
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %8) #10
  %61 = load ptr, ptr %0, align 8
  %62 = getelementptr inbounds nuw i8, ptr %61, i64 4
  %63 = load i32, ptr %62, align 4
  %64 = sext i32 %63 to i64
  %65 = getelementptr inbounds i8, ptr %0, i64 %64
  tail call void @llvm.experimental.noalias.scope.decl(metadata !24)
  %66 = getelementptr inbounds nuw i8, ptr %65, i64 64
  %67 = load ptr, ptr %66, align 8, !noalias !24
  %68 = getelementptr inbounds nuw i8, ptr %8, i64 8
  %69 = getelementptr inbounds nuw i8, ptr %67, i64 8
  %70 = load ptr, ptr %69, align 8, !noalias !24
  store ptr %70, ptr %68, align 8, !alias.scope !24
  %71 = load ptr, ptr %70, align 8, !noalias !24
  %72 = getelementptr inbounds nuw i8, ptr %71, i64 8
  %73 = load ptr, ptr %72, align 8, !noalias !24
  tail call void %73(ptr noundef nonnull align 8 dereferenceable(16) %70) #10, !noalias !24
  %74 = invoke noundef nonnull align 8 dereferenceable(16) ptr @"??$use_facet@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z"(ptr noundef nonnull align 8 dereferenceable(16) %8)
          to label %75 unwind label %106

75:                                               ; preds = %60
  %76 = load ptr, ptr %68, align 8
  %77 = icmp eq ptr %76, null
  br i1 %77, label %88, label %78

78:                                               ; preds = %75
  %79 = load ptr, ptr %76, align 8
  %80 = getelementptr inbounds nuw i8, ptr %79, i64 16
  %81 = load ptr, ptr %80, align 8
  %82 = call noundef ptr %81(ptr noundef nonnull align 8 dereferenceable(16) %76) #10
  %83 = icmp eq ptr %82, null
  br i1 %83, label %88, label %84

84:                                               ; preds = %78
  %85 = load ptr, ptr %82, align 8
  %86 = load ptr, ptr %85, align 8
  %87 = call noundef ptr %86(ptr noundef nonnull align 8 dereferenceable(8) %82, i32 noundef 1) #10
  br label %88

88:                                               ; preds = %75, %78, %84
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %8) #10
  %89 = load ptr, ptr %0, align 8
  %90 = getelementptr inbounds nuw i8, ptr %89, i64 4
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %9) #10
  %91 = load i32, ptr %90, align 4
  %92 = sext i32 %91 to i64
  %93 = getelementptr inbounds i8, ptr %0, i64 %92
  %94 = getelementptr inbounds nuw i8, ptr %93, i64 88
  %95 = load i8, ptr %94, align 8
  %96 = getelementptr inbounds nuw i8, ptr %93, i64 72
  %97 = load ptr, ptr %96, align 8
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %6)
  store i8 0, ptr %6, align 8, !noalias !27
  %98 = getelementptr inbounds nuw i8, ptr %6, i64 8
  store ptr %97, ptr %98, align 8, !noalias !27
  %99 = load ptr, ptr %74, align 8, !noalias !27
  %100 = getelementptr inbounds nuw i8, ptr %99, i64 72
  %101 = load ptr, ptr %100, align 8, !noalias !27
  invoke void %101(ptr noundef nonnull align 8 dereferenceable(16) %74, ptr dead_on_unwind nonnull writable sret(%"class.std::ostreambuf_iterator") align 8 %9, ptr dead_on_return noundef nonnull %6, ptr noundef nonnull align 8 dereferenceable(72) %93, i8 noundef %95, i32 noundef %1)
          to label %102 unwind label %121

102:                                              ; preds = %88
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %6)
  %103 = load i8, ptr %9, align 8, !range !18, !noundef !19
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %9) #10
  %104 = shl nuw nsw i8 %103, 2
  %105 = zext nneg i8 %104 to i32
  br label %131

106:                                              ; preds = %60
  %107 = cleanuppad within none []
  %108 = load ptr, ptr %68, align 8
  %109 = icmp eq ptr %108, null
  br i1 %109, label %120, label %110

110:                                              ; preds = %106
  %111 = load ptr, ptr %108, align 8
  %112 = getelementptr inbounds nuw i8, ptr %111, i64 16
  %113 = load ptr, ptr %112, align 8
  %114 = call noundef ptr %113(ptr noundef nonnull align 8 dereferenceable(16) %108) #10 [ "funclet"(token %107) ]
  %115 = icmp eq ptr %114, null
  br i1 %115, label %120, label %116

116:                                              ; preds = %110
  %117 = load ptr, ptr %114, align 8
  %118 = load ptr, ptr %117, align 8
  %119 = call noundef ptr %118(ptr noundef nonnull align 8 dereferenceable(8) %114, i32 noundef 1) #10 [ "funclet"(token %107) ]
  br label %120

120:                                              ; preds = %106, %110, %116
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %8) #10
  cleanupret from %107 unwind label %182

121:                                              ; preds = %88
  %122 = catchswitch within none [label %123] unwind label %182

123:                                              ; preds = %121
  %124 = catchpad within %122 [ptr null, i32 64, ptr null]
  %125 = load ptr, ptr %0, align 8
  %126 = getelementptr inbounds nuw i8, ptr %125, i64 4
  %127 = load i32, ptr %126, align 4
  %128 = sext i32 %127 to i64
  %129 = getelementptr inbounds i8, ptr %0, i64 %128
  invoke void @"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"(ptr noundef nonnull align 8 dereferenceable(96) %129, i32 noundef 4, i1 noundef zeroext true) [ "funclet"(token %124) ]
          to label %130 unwind label %182

130:                                              ; preds = %123
  catchret from %124 to label %131

131:                                              ; preds = %102, %130, %56
  %132 = phi i32 [ 0, %56 ], [ 0, %130 ], [ %105, %102 ]
  %133 = load ptr, ptr %0, align 8
  %134 = getelementptr inbounds nuw i8, ptr %133, i64 4
  %135 = load i32, ptr %134, align 4
  %136 = sext i32 %135 to i64
  %137 = getelementptr inbounds i8, ptr %0, i64 %136
  %138 = getelementptr inbounds nuw i8, ptr %137, i64 16
  %139 = load i32, ptr %138, align 8
  %140 = or i32 %139, %132
  %141 = getelementptr inbounds nuw i8, ptr %137, i64 72
  %142 = load ptr, ptr %141, align 8
  %143 = icmp eq ptr %142, null
  %144 = select i1 %143, i32 4, i32 0
  call void @llvm.lifetime.start.p0(i64 40, ptr nonnull %4)
  %145 = and i32 %140, 23
  %146 = or i32 %144, %145
  store i32 %146, ptr %138, align 8
  %147 = getelementptr inbounds nuw i8, ptr %137, i64 20
  %148 = load i32, ptr %147, align 4
  %149 = and i32 %148, %146
  %150 = icmp eq i32 %149, 0
  br i1 %150, label %161, label %151

151:                                              ; preds = %131
  %152 = and i32 %149, 4
  %153 = icmp eq i32 %152, 0
  %154 = and i32 %149, 2
  %155 = icmp eq i32 %154, 0
  %156 = select i1 %155, ptr @"??_C@_0BF@OOHOMBOF@ios_base?3?3eofbit?5set?$AA@", ptr @"??_C@_0BG@FMKFHCIL@ios_base?3?3failbit?5set?$AA@"
  %157 = select i1 %153, ptr %156, ptr @"??_C@_0BF@PHHKMMFD@ios_base?3?3badbit?5set?$AA@"
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %5) #10
  call void @"?make_error_code@std@@YA?AVerror_code@1@W4io_errc@1@@Z"(ptr dead_on_unwind nonnull writable sret(%"class.std::error_code") align 8 %5, i32 noundef 1) #10
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %3)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %3, ptr noundef nonnull align 8 dereferenceable(16) %5, i64 16, i1 false)
  %158 = invoke noundef ptr @"??0system_error@std@@QEAA@Verror_code@1@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(40) %4, ptr dead_on_return noundef nonnull %3, ptr noundef nonnull %157)
          to label %159 unwind label %182

159:                                              ; preds = %151
  store ptr @"??_7failure@ios_base@std@@6B@", ptr %4, align 8
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %3)
  invoke void @_CxxThrowException(ptr nonnull %4, ptr nonnull @"_TI5?AVfailure@ios_base@std@@") #29
          to label %160 unwind label %182

160:                                              ; preds = %159
  unreachable

161:                                              ; preds = %131
  call void @llvm.lifetime.end.p0(i64 40, ptr nonnull %4)
  %162 = call noundef zeroext i1 @"?uncaught_exception@std@@YA_NXZ"() #10
  br i1 %162, label %165, label %163

163:                                              ; preds = %161
  %164 = load ptr, ptr %7, align 8, !nonnull !19, !align !20
  call void @"?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ"(ptr noundef nonnull align 8 dereferenceable(8) %164) #10
  br label %165

165:                                              ; preds = %163, %161
  %166 = load ptr, ptr %7, align 8, !nonnull !19, !align !20
  %167 = load ptr, ptr %166, align 8
  %168 = getelementptr inbounds nuw i8, ptr %167, i64 4
  %169 = load i32, ptr %168, align 4
  %170 = sext i32 %169 to i64
  %171 = getelementptr inbounds i8, ptr %166, i64 %170
  %172 = getelementptr inbounds nuw i8, ptr %171, i64 72
  %173 = load ptr, ptr %172, align 8
  %174 = icmp eq ptr %173, null
  br i1 %174, label %181, label %175

175:                                              ; preds = %165
  %176 = load ptr, ptr %173, align 8
  %177 = getelementptr inbounds nuw i8, ptr %176, i64 16
  %178 = load ptr, ptr %177, align 8
  invoke void %178(ptr noundef nonnull align 8 dereferenceable(104) %173)
          to label %181 unwind label %179

179:                                              ; preds = %175
  %180 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %180) ]
  unreachable

181:                                              ; preds = %165, %175
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %7) #10
  ret ptr %0

182:                                              ; preds = %151, %159, %121, %123, %120
  %183 = cleanuppad within none []
  call void @"??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %7) #10 [ "funclet"(token %183) ]
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %7) #10
  cleanupret from %183 unwind to caller
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(8) ptr @"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@PEBX@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0, ptr noundef %1) local_unnamed_addr #2 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %3 = alloca %"class.std::error_code", align 8
  %4 = alloca %"class.std::ios_base::failure", align 8
  %5 = alloca %"class.std::error_code", align 8
  %6 = alloca %"class.std::ostreambuf_iterator", align 8
  %7 = alloca %"class.std::basic_ostream<char>::sentry", align 8
  %8 = alloca %"class.std::locale", align 8
  %9 = alloca %"class.std::ostreambuf_iterator", align 8
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %7) #10
  store ptr %0, ptr %7, align 8
  %10 = load ptr, ptr %0, align 8
  %11 = getelementptr inbounds nuw i8, ptr %10, i64 4
  %12 = load i32, ptr %11, align 4
  %13 = sext i32 %12 to i64
  %14 = getelementptr inbounds i8, ptr %0, i64 %13
  %15 = getelementptr inbounds nuw i8, ptr %14, i64 72
  %16 = load ptr, ptr %15, align 8
  %17 = icmp eq ptr %16, null
  br i1 %17, label %22, label %18

18:                                               ; preds = %2
  %19 = load ptr, ptr %16, align 8
  %20 = getelementptr inbounds nuw i8, ptr %19, i64 8
  %21 = load ptr, ptr %20, align 8
  tail call void %21(ptr noundef nonnull align 8 dereferenceable(104) %16)
  br label %22

22:                                               ; preds = %18, %2
  %23 = load ptr, ptr %0, align 8
  %24 = getelementptr inbounds nuw i8, ptr %23, i64 4
  %25 = load i32, ptr %24, align 4
  %26 = sext i32 %25 to i64
  %27 = getelementptr inbounds i8, ptr %0, i64 %26
  %28 = getelementptr inbounds nuw i8, ptr %27, i64 16
  %29 = load i32, ptr %28, align 8
  %30 = icmp eq i32 %29, 0
  br i1 %30, label %33, label %31

31:                                               ; preds = %22
  %32 = getelementptr inbounds nuw i8, ptr %7, i64 8
  store i8 0, ptr %32, align 8
  br label %56

33:                                               ; preds = %22
  %34 = getelementptr inbounds nuw i8, ptr %27, i64 80
  %35 = load ptr, ptr %34, align 8
  %36 = icmp eq ptr %35, null
  %37 = icmp eq ptr %35, %0
  %38 = or i1 %36, %37
  br i1 %38, label %39, label %41

39:                                               ; preds = %33
  %40 = getelementptr inbounds nuw i8, ptr %7, i64 8
  store i8 1, ptr %40, align 8
  br label %56

41:                                               ; preds = %33
  %42 = invoke noundef nonnull align 8 dereferenceable(8) ptr @"?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %35)
          to label %43 unwind label %54

43:                                               ; preds = %41
  %44 = load ptr, ptr %0, align 8
  %45 = getelementptr inbounds nuw i8, ptr %44, i64 4
  %46 = load i32, ptr %45, align 4
  %47 = sext i32 %46 to i64
  %48 = getelementptr inbounds i8, ptr %0, i64 %47
  %49 = getelementptr inbounds nuw i8, ptr %48, i64 16
  %50 = load i32, ptr %49, align 8
  %51 = icmp eq i32 %50, 0
  %52 = getelementptr inbounds nuw i8, ptr %7, i64 8
  %53 = zext i1 %51 to i8
  store i8 %53, ptr %52, align 8
  br label %56

54:                                               ; preds = %41
  %55 = cleanuppad within none []
  call void @"??1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %7) #10 [ "funclet"(token %55) ]
  cleanupret from %55 unwind to caller

56:                                               ; preds = %31, %39, %43
  %57 = getelementptr inbounds nuw i8, ptr %7, i64 8
  %58 = load i8, ptr %57, align 8, !range !18, !noundef !19
  %59 = trunc nuw i8 %58 to i1
  br i1 %59, label %60, label %131

60:                                               ; preds = %56
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %8) #10
  %61 = load ptr, ptr %0, align 8
  %62 = getelementptr inbounds nuw i8, ptr %61, i64 4
  %63 = load i32, ptr %62, align 4
  %64 = sext i32 %63 to i64
  %65 = getelementptr inbounds i8, ptr %0, i64 %64
  tail call void @llvm.experimental.noalias.scope.decl(metadata !30)
  %66 = getelementptr inbounds nuw i8, ptr %65, i64 64
  %67 = load ptr, ptr %66, align 8, !noalias !30
  %68 = getelementptr inbounds nuw i8, ptr %8, i64 8
  %69 = getelementptr inbounds nuw i8, ptr %67, i64 8
  %70 = load ptr, ptr %69, align 8, !noalias !30
  store ptr %70, ptr %68, align 8, !alias.scope !30
  %71 = load ptr, ptr %70, align 8, !noalias !30
  %72 = getelementptr inbounds nuw i8, ptr %71, i64 8
  %73 = load ptr, ptr %72, align 8, !noalias !30
  tail call void %73(ptr noundef nonnull align 8 dereferenceable(16) %70) #10, !noalias !30
  %74 = invoke noundef nonnull align 8 dereferenceable(16) ptr @"??$use_facet@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z"(ptr noundef nonnull align 8 dereferenceable(16) %8)
          to label %75 unwind label %106

75:                                               ; preds = %60
  %76 = load ptr, ptr %68, align 8
  %77 = icmp eq ptr %76, null
  br i1 %77, label %88, label %78

78:                                               ; preds = %75
  %79 = load ptr, ptr %76, align 8
  %80 = getelementptr inbounds nuw i8, ptr %79, i64 16
  %81 = load ptr, ptr %80, align 8
  %82 = call noundef ptr %81(ptr noundef nonnull align 8 dereferenceable(16) %76) #10
  %83 = icmp eq ptr %82, null
  br i1 %83, label %88, label %84

84:                                               ; preds = %78
  %85 = load ptr, ptr %82, align 8
  %86 = load ptr, ptr %85, align 8
  %87 = call noundef ptr %86(ptr noundef nonnull align 8 dereferenceable(8) %82, i32 noundef 1) #10
  br label %88

88:                                               ; preds = %75, %78, %84
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %8) #10
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %9) #10
  %89 = load ptr, ptr %0, align 8
  %90 = getelementptr inbounds nuw i8, ptr %89, i64 4
  %91 = load i32, ptr %90, align 4
  %92 = sext i32 %91 to i64
  %93 = getelementptr inbounds i8, ptr %0, i64 %92
  %94 = getelementptr inbounds nuw i8, ptr %93, i64 88
  %95 = load i8, ptr %94, align 8
  %96 = getelementptr inbounds nuw i8, ptr %93, i64 72
  %97 = load ptr, ptr %96, align 8
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %6)
  store i8 0, ptr %6, align 8, !noalias !33
  %98 = getelementptr inbounds nuw i8, ptr %6, i64 8
  store ptr %97, ptr %98, align 8, !noalias !33
  %99 = load ptr, ptr %74, align 8, !noalias !33
  %100 = getelementptr inbounds nuw i8, ptr %99, i64 24
  %101 = load ptr, ptr %100, align 8, !noalias !33
  invoke void %101(ptr noundef nonnull align 8 dereferenceable(16) %74, ptr dead_on_unwind nonnull writable sret(%"class.std::ostreambuf_iterator") align 8 %9, ptr dead_on_return noundef nonnull %6, ptr noundef nonnull align 8 dereferenceable(72) %93, i8 noundef %95, ptr noundef %1)
          to label %102 unwind label %121

102:                                              ; preds = %88
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %6)
  %103 = load i8, ptr %9, align 8, !range !18, !noundef !19
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %9) #10
  %104 = shl nuw nsw i8 %103, 2
  %105 = zext nneg i8 %104 to i32
  br label %131

106:                                              ; preds = %60
  %107 = cleanuppad within none []
  %108 = load ptr, ptr %68, align 8
  %109 = icmp eq ptr %108, null
  br i1 %109, label %120, label %110

110:                                              ; preds = %106
  %111 = load ptr, ptr %108, align 8
  %112 = getelementptr inbounds nuw i8, ptr %111, i64 16
  %113 = load ptr, ptr %112, align 8
  %114 = call noundef ptr %113(ptr noundef nonnull align 8 dereferenceable(16) %108) #10 [ "funclet"(token %107) ]
  %115 = icmp eq ptr %114, null
  br i1 %115, label %120, label %116

116:                                              ; preds = %110
  %117 = load ptr, ptr %114, align 8
  %118 = load ptr, ptr %117, align 8
  %119 = call noundef ptr %118(ptr noundef nonnull align 8 dereferenceable(8) %114, i32 noundef 1) #10 [ "funclet"(token %107) ]
  br label %120

120:                                              ; preds = %106, %110, %116
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %8) #10
  cleanupret from %107 unwind label %182

121:                                              ; preds = %88
  %122 = catchswitch within none [label %123] unwind label %182

123:                                              ; preds = %121
  %124 = catchpad within %122 [ptr null, i32 64, ptr null]
  %125 = load ptr, ptr %0, align 8
  %126 = getelementptr inbounds nuw i8, ptr %125, i64 4
  %127 = load i32, ptr %126, align 4
  %128 = sext i32 %127 to i64
  %129 = getelementptr inbounds i8, ptr %0, i64 %128
  invoke void @"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"(ptr noundef nonnull align 8 dereferenceable(96) %129, i32 noundef 4, i1 noundef zeroext true) [ "funclet"(token %124) ]
          to label %130 unwind label %182

130:                                              ; preds = %123
  catchret from %124 to label %131

131:                                              ; preds = %102, %130, %56
  %132 = phi i32 [ 0, %56 ], [ 0, %130 ], [ %105, %102 ]
  %133 = load ptr, ptr %0, align 8
  %134 = getelementptr inbounds nuw i8, ptr %133, i64 4
  %135 = load i32, ptr %134, align 4
  %136 = sext i32 %135 to i64
  %137 = getelementptr inbounds i8, ptr %0, i64 %136
  %138 = getelementptr inbounds nuw i8, ptr %137, i64 16
  %139 = load i32, ptr %138, align 8
  %140 = or i32 %139, %132
  %141 = getelementptr inbounds nuw i8, ptr %137, i64 72
  %142 = load ptr, ptr %141, align 8
  %143 = icmp eq ptr %142, null
  %144 = select i1 %143, i32 4, i32 0
  call void @llvm.lifetime.start.p0(i64 40, ptr nonnull %4)
  %145 = and i32 %140, 23
  %146 = or i32 %144, %145
  store i32 %146, ptr %138, align 8
  %147 = getelementptr inbounds nuw i8, ptr %137, i64 20
  %148 = load i32, ptr %147, align 4
  %149 = and i32 %148, %146
  %150 = icmp eq i32 %149, 0
  br i1 %150, label %161, label %151

151:                                              ; preds = %131
  %152 = and i32 %149, 4
  %153 = icmp eq i32 %152, 0
  %154 = and i32 %149, 2
  %155 = icmp eq i32 %154, 0
  %156 = select i1 %155, ptr @"??_C@_0BF@OOHOMBOF@ios_base?3?3eofbit?5set?$AA@", ptr @"??_C@_0BG@FMKFHCIL@ios_base?3?3failbit?5set?$AA@"
  %157 = select i1 %153, ptr %156, ptr @"??_C@_0BF@PHHKMMFD@ios_base?3?3badbit?5set?$AA@"
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %5) #10
  call void @"?make_error_code@std@@YA?AVerror_code@1@W4io_errc@1@@Z"(ptr dead_on_unwind nonnull writable sret(%"class.std::error_code") align 8 %5, i32 noundef 1) #10
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %3)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %3, ptr noundef nonnull align 8 dereferenceable(16) %5, i64 16, i1 false)
  %158 = invoke noundef ptr @"??0system_error@std@@QEAA@Verror_code@1@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(40) %4, ptr dead_on_return noundef nonnull %3, ptr noundef nonnull %157)
          to label %159 unwind label %182

159:                                              ; preds = %151
  store ptr @"??_7failure@ios_base@std@@6B@", ptr %4, align 8
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %3)
  invoke void @_CxxThrowException(ptr nonnull %4, ptr nonnull @"_TI5?AVfailure@ios_base@std@@") #29
          to label %160 unwind label %182

160:                                              ; preds = %159
  unreachable

161:                                              ; preds = %131
  call void @llvm.lifetime.end.p0(i64 40, ptr nonnull %4)
  %162 = call noundef zeroext i1 @"?uncaught_exception@std@@YA_NXZ"() #10
  br i1 %162, label %165, label %163

163:                                              ; preds = %161
  %164 = load ptr, ptr %7, align 8, !nonnull !19, !align !20
  call void @"?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ"(ptr noundef nonnull align 8 dereferenceable(8) %164) #10
  br label %165

165:                                              ; preds = %163, %161
  %166 = load ptr, ptr %7, align 8, !nonnull !19, !align !20
  %167 = load ptr, ptr %166, align 8
  %168 = getelementptr inbounds nuw i8, ptr %167, i64 4
  %169 = load i32, ptr %168, align 4
  %170 = sext i32 %169 to i64
  %171 = getelementptr inbounds i8, ptr %166, i64 %170
  %172 = getelementptr inbounds nuw i8, ptr %171, i64 72
  %173 = load ptr, ptr %172, align 8
  %174 = icmp eq ptr %173, null
  br i1 %174, label %181, label %175

175:                                              ; preds = %165
  %176 = load ptr, ptr %173, align 8
  %177 = getelementptr inbounds nuw i8, ptr %176, i64 16
  %178 = load ptr, ptr %177, align 8
  invoke void %178(ptr noundef nonnull align 8 dereferenceable(104) %173)
          to label %181 unwind label %179

179:                                              ; preds = %175
  %180 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %180) ]
  unreachable

181:                                              ; preds = %165, %175
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %7) #10
  ret ptr %0

182:                                              ; preds = %151, %159, %120, %121, %123
  %183 = cleanuppad within none []
  call void @"??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %7) #10 [ "funclet"(token %183) ]
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %7) #10
  cleanupret from %183 unwind to caller
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #3

; Function Attrs: nobuiltin allocsize(0)
declare dso_local noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef) local_unnamed_addr #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr captures(none)) #1

; Function Attrs: nobuiltin nounwind
declare dso_local void @"??3@YAXPEAX_K@Z"(ptr noundef, i64 noundef) local_unnamed_addr #5

; Function Attrs: uwtable
define linkonce_odr dso_local void @"??__E?id@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A@@YAXXZ"() #6 comdat {
  store i64 0, ptr @"?id@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A", align 8
  ret void
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @"??__E?id@?$numpunct@D@std@@2V0locale@2@A@@YAXXZ"() #6 comdat {
  store i64 0, ptr @"?id@?$numpunct@D@std@@2V0locale@2@A", align 8
  ret void
}

declare dso_local i32 @__CxxFrameHandler3(...)

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"(ptr noundef nonnull align 8 dereferenceable(96) %0, i32 noundef %1, i1 noundef zeroext %2) local_unnamed_addr #2 comdat align 2 {
  %4 = alloca %"class.std::ios_base::failure", align 8
  %5 = alloca %"class.std::error_code", align 8
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %7 = load i32, ptr %6, align 8
  %8 = or i32 %7, %1
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 72
  %10 = load ptr, ptr %9, align 8
  %11 = icmp eq ptr %10, null
  %12 = select i1 %11, i32 4, i32 0
  call void @llvm.lifetime.start.p0(i64 40, ptr nonnull %4)
  %13 = and i32 %8, 23
  %14 = or i32 %12, %13
  store i32 %14, ptr %6, align 8
  %15 = getelementptr inbounds nuw i8, ptr %0, i64 20
  %16 = load i32, ptr %15, align 4
  %17 = and i32 %16, %14
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %29, label %19

19:                                               ; preds = %3
  br i1 %2, label %20, label %21

20:                                               ; preds = %19
  tail call void @_CxxThrowException(ptr null, ptr null) #29
  unreachable

21:                                               ; preds = %19
  %22 = and i32 %17, 4
  %23 = icmp eq i32 %22, 0
  %24 = and i32 %17, 2
  %25 = icmp eq i32 %24, 0
  %26 = select i1 %25, ptr @"??_C@_0BF@OOHOMBOF@ios_base?3?3eofbit?5set?$AA@", ptr @"??_C@_0BG@FMKFHCIL@ios_base?3?3failbit?5set?$AA@"
  %27 = select i1 %23, ptr %26, ptr @"??_C@_0BF@PHHKMMFD@ios_base?3?3badbit?5set?$AA@"
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %5) #10
  call void @"?make_error_code@std@@YA?AVerror_code@1@W4io_errc@1@@Z"(ptr dead_on_unwind nonnull writable sret(%"class.std::error_code") align 8 %5, i32 noundef 1) #10
  %28 = call noundef ptr @"??0failure@ios_base@std@@QEAA@PEBDAEBVerror_code@2@@Z"(ptr noundef nonnull align 8 dereferenceable(40) %4, ptr noundef nonnull %27, ptr noundef nonnull align 8 dereferenceable(16) %5)
  call void @_CxxThrowException(ptr nonnull %4, ptr nonnull @"_TI5?AVfailure@ios_base@std@@") #29
  unreachable

29:                                               ; preds = %3
  call void @llvm.lifetime.end.p0(i64 40, ptr nonnull %4)
  ret void
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @"??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %0) unnamed_addr #7 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %2 = tail call noundef zeroext i1 @"?uncaught_exception@std@@YA_NXZ"() #10
  br i1 %2, label %5, label %3

3:                                                ; preds = %1
  %4 = load ptr, ptr %0, align 8, !nonnull !19, !align !20
  tail call void @"?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ"(ptr noundef nonnull align 8 dereferenceable(8) %4) #10
  br label %5

5:                                                ; preds = %3, %1
  %6 = load ptr, ptr %0, align 8, !nonnull !19, !align !20
  %7 = load ptr, ptr %6, align 8
  %8 = getelementptr inbounds nuw i8, ptr %7, i64 4
  %9 = load i32, ptr %8, align 4
  %10 = sext i32 %9 to i64
  %11 = getelementptr inbounds i8, ptr %6, i64 %10
  %12 = getelementptr inbounds nuw i8, ptr %11, i64 72
  %13 = load ptr, ptr %12, align 8
  %14 = icmp eq ptr %13, null
  br i1 %14, label %21, label %15

15:                                               ; preds = %5
  %16 = load ptr, ptr %13, align 8
  %17 = getelementptr inbounds nuw i8, ptr %16, i64 16
  %18 = load ptr, ptr %17, align 8
  invoke void %18(ptr noundef nonnull align 8 dereferenceable(104) %13)
          to label %21 unwind label %19

19:                                               ; preds = %15
  %20 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %20) ]
  unreachable

21:                                               ; preds = %5, %15
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare dso_local i64 @strlen(ptr noundef captures(none)) local_unnamed_addr #8

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(8) ptr @"?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) local_unnamed_addr #2 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %2 = alloca %"class.std::error_code", align 8
  %3 = alloca %"class.std::ios_base::failure", align 8
  %4 = alloca %"class.std::error_code", align 8
  %5 = alloca %"class.std::basic_ostream<char>::sentry", align 8
  %6 = load ptr, ptr %0, align 8
  %7 = getelementptr inbounds nuw i8, ptr %6, i64 4
  %8 = load i32, ptr %7, align 4
  %9 = sext i32 %8 to i64
  %10 = getelementptr inbounds i8, ptr %0, i64 %9
  %11 = getelementptr inbounds nuw i8, ptr %10, i64 72
  %12 = load ptr, ptr %11, align 8
  %13 = icmp eq ptr %12, null
  br i1 %13, label %135, label %14

14:                                               ; preds = %1
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %5) #10
  store ptr %0, ptr %5, align 8
  %15 = load i32, ptr %7, align 4
  %16 = sext i32 %15 to i64
  %17 = getelementptr inbounds i8, ptr %0, i64 %16
  %18 = getelementptr inbounds nuw i8, ptr %17, i64 72
  %19 = load ptr, ptr %18, align 8
  %20 = icmp eq ptr %19, null
  br i1 %20, label %25, label %21

21:                                               ; preds = %14
  %22 = load ptr, ptr %19, align 8
  %23 = getelementptr inbounds nuw i8, ptr %22, i64 8
  %24 = load ptr, ptr %23, align 8
  tail call void %24(ptr noundef nonnull align 8 dereferenceable(104) %19)
  br label %25

25:                                               ; preds = %21, %14
  %26 = load ptr, ptr %0, align 8
  %27 = getelementptr inbounds nuw i8, ptr %26, i64 4
  %28 = load i32, ptr %27, align 4
  %29 = sext i32 %28 to i64
  %30 = getelementptr inbounds i8, ptr %0, i64 %29
  %31 = getelementptr inbounds nuw i8, ptr %30, i64 16
  %32 = load i32, ptr %31, align 8
  %33 = icmp eq i32 %32, 0
  br i1 %33, label %36, label %34

34:                                               ; preds = %25
  %35 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store i8 0, ptr %35, align 8
  br label %59

36:                                               ; preds = %25
  %37 = getelementptr inbounds nuw i8, ptr %30, i64 80
  %38 = load ptr, ptr %37, align 8
  %39 = icmp eq ptr %38, null
  %40 = icmp eq ptr %38, %0
  %41 = or i1 %39, %40
  br i1 %41, label %42, label %44

42:                                               ; preds = %36
  %43 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store i8 1, ptr %43, align 8
  br label %59

44:                                               ; preds = %36
  %45 = invoke noundef nonnull align 8 dereferenceable(8) ptr @"?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %38)
          to label %46 unwind label %57

46:                                               ; preds = %44
  %47 = load ptr, ptr %0, align 8
  %48 = getelementptr inbounds nuw i8, ptr %47, i64 4
  %49 = load i32, ptr %48, align 4
  %50 = sext i32 %49 to i64
  %51 = getelementptr inbounds i8, ptr %0, i64 %50
  %52 = getelementptr inbounds nuw i8, ptr %51, i64 16
  %53 = load i32, ptr %52, align 8
  %54 = icmp eq i32 %53, 0
  %55 = getelementptr inbounds nuw i8, ptr %5, i64 8
  %56 = zext i1 %54 to i8
  store i8 %56, ptr %55, align 8
  br label %59

57:                                               ; preds = %44
  %58 = cleanuppad within none []
  call void @"??1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %5) #10 [ "funclet"(token %58) ]
  cleanupret from %58 unwind to caller

59:                                               ; preds = %34, %42, %46
  %60 = getelementptr inbounds nuw i8, ptr %5, i64 8
  %61 = load i8, ptr %60, align 8, !range !18, !noundef !19
  %62 = trunc nuw i8 %61 to i1
  br i1 %62, label %63, label %114

63:                                               ; preds = %59
  %64 = load ptr, ptr %12, align 8
  %65 = getelementptr inbounds nuw i8, ptr %64, i64 104
  %66 = load ptr, ptr %65, align 8
  %67 = invoke noundef i32 %66(ptr noundef nonnull align 8 dereferenceable(104) %12)
          to label %109 unwind label %68

68:                                               ; preds = %63
  %69 = catchswitch within none [label %70] unwind label %112

70:                                               ; preds = %68
  %71 = catchpad within %69 [ptr null, i32 64, ptr null]
  %72 = load ptr, ptr %0, align 8
  %73 = getelementptr inbounds nuw i8, ptr %72, i64 4
  %74 = load i32, ptr %73, align 4
  %75 = sext i32 %74 to i64
  %76 = getelementptr inbounds i8, ptr %0, i64 %75
  invoke void @"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"(ptr noundef nonnull align 8 dereferenceable(96) %76, i32 noundef 4, i1 noundef zeroext true) [ "funclet"(token %71) ]
          to label %77 unwind label %112

77:                                               ; preds = %70
  catchret from %71 to label %78

78:                                               ; preds = %109, %77
  %79 = phi i32 [ 0, %77 ], [ %111, %109 ]
  %80 = load ptr, ptr %0, align 8
  %81 = getelementptr inbounds nuw i8, ptr %80, i64 4
  %82 = load i32, ptr %81, align 4
  %83 = sext i32 %82 to i64
  %84 = getelementptr inbounds i8, ptr %0, i64 %83
  %85 = getelementptr inbounds nuw i8, ptr %84, i64 16
  %86 = load i32, ptr %85, align 8
  %87 = getelementptr inbounds nuw i8, ptr %84, i64 72
  %88 = load ptr, ptr %87, align 8
  %89 = icmp eq ptr %88, null
  %90 = select i1 %89, i32 4, i32 0
  call void @llvm.lifetime.start.p0(i64 40, ptr nonnull %3)
  %91 = and i32 %86, 23
  %92 = or i32 %91, %79
  %93 = or i32 %92, %90
  store i32 %93, ptr %85, align 8
  %94 = getelementptr inbounds nuw i8, ptr %84, i64 20
  %95 = load i32, ptr %94, align 4
  %96 = and i32 %95, %93
  %97 = icmp eq i32 %96, 0
  br i1 %97, label %108, label %98

98:                                               ; preds = %78
  %99 = and i32 %96, 4
  %100 = icmp eq i32 %99, 0
  %101 = and i32 %96, 2
  %102 = icmp eq i32 %101, 0
  %103 = select i1 %102, ptr @"??_C@_0BF@OOHOMBOF@ios_base?3?3eofbit?5set?$AA@", ptr @"??_C@_0BG@FMKFHCIL@ios_base?3?3failbit?5set?$AA@"
  %104 = select i1 %100, ptr %103, ptr @"??_C@_0BF@PHHKMMFD@ios_base?3?3badbit?5set?$AA@"
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %4) #10
  call void @"?make_error_code@std@@YA?AVerror_code@1@W4io_errc@1@@Z"(ptr dead_on_unwind nonnull writable sret(%"class.std::error_code") align 8 %4, i32 noundef 1) #10
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %2)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %2, ptr noundef nonnull align 8 dereferenceable(16) %4, i64 16, i1 false)
  %105 = invoke noundef ptr @"??0system_error@std@@QEAA@Verror_code@1@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(40) %3, ptr dead_on_return noundef nonnull %2, ptr noundef nonnull %104)
          to label %106 unwind label %112

106:                                              ; preds = %98
  store ptr @"??_7failure@ios_base@std@@6B@", ptr %3, align 8
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %2)
  invoke void @_CxxThrowException(ptr nonnull %3, ptr nonnull @"_TI5?AVfailure@ios_base@std@@") #29
          to label %107 unwind label %112

107:                                              ; preds = %106
  unreachable

108:                                              ; preds = %78
  call void @llvm.lifetime.end.p0(i64 40, ptr nonnull %3)
  br label %114

109:                                              ; preds = %63
  %110 = icmp eq i32 %67, -1
  %111 = select i1 %110, i32 4, i32 0
  br label %78

112:                                              ; preds = %98, %106, %70, %68
  %113 = cleanuppad within none []
  call void @"??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %5) #10 [ "funclet"(token %113) ]
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %5) #10
  cleanupret from %113 unwind to caller

114:                                              ; preds = %108, %59
  %115 = tail call noundef zeroext i1 @"?uncaught_exception@std@@YA_NXZ"() #10
  br i1 %115, label %118, label %116

116:                                              ; preds = %114
  %117 = load ptr, ptr %5, align 8, !nonnull !19, !align !20
  tail call void @"?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ"(ptr noundef nonnull align 8 dereferenceable(8) %117) #10
  br label %118

118:                                              ; preds = %116, %114
  %119 = load ptr, ptr %5, align 8, !nonnull !19, !align !20
  %120 = load ptr, ptr %119, align 8
  %121 = getelementptr inbounds nuw i8, ptr %120, i64 4
  %122 = load i32, ptr %121, align 4
  %123 = sext i32 %122 to i64
  %124 = getelementptr inbounds i8, ptr %119, i64 %123
  %125 = getelementptr inbounds nuw i8, ptr %124, i64 72
  %126 = load ptr, ptr %125, align 8
  %127 = icmp eq ptr %126, null
  br i1 %127, label %134, label %128

128:                                              ; preds = %118
  %129 = load ptr, ptr %126, align 8
  %130 = getelementptr inbounds nuw i8, ptr %129, i64 16
  %131 = load ptr, ptr %130, align 8
  invoke void %131(ptr noundef nonnull align 8 dereferenceable(104) %126)
          to label %134 unwind label %132

132:                                              ; preds = %128
  %133 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %133) ]
  unreachable

134:                                              ; preds = %118, %128
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %5) #10
  br label %135

135:                                              ; preds = %134, %1
  ret ptr %0
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @"??1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) unnamed_addr #7 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %2 = load ptr, ptr %0, align 8, !nonnull !19, !align !20
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw i8, ptr %3, i64 4
  %5 = load i32, ptr %4, align 4
  %6 = sext i32 %5 to i64
  %7 = getelementptr inbounds i8, ptr %2, i64 %6
  %8 = getelementptr inbounds nuw i8, ptr %7, i64 72
  %9 = load ptr, ptr %8, align 8
  %10 = icmp eq ptr %9, null
  br i1 %10, label %15, label %11

11:                                               ; preds = %1
  %12 = load ptr, ptr %9, align 8
  %13 = getelementptr inbounds nuw i8, ptr %12, i64 16
  %14 = load ptr, ptr %13, align 8
  invoke void %14(ptr noundef nonnull align 8 dereferenceable(104) %9)
          to label %15 unwind label %16

15:                                               ; preds = %11, %1
  ret void

16:                                               ; preds = %11
  %17 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %17) ]
  unreachable
}

declare dso_local void @__std_terminate() local_unnamed_addr

declare dso_local void @_CxxThrowException(ptr, ptr) local_unnamed_addr

; Function Attrs: inlinehint mustprogress nounwind uwtable
define linkonce_odr dso_local void @"?make_error_code@std@@YA?AVerror_code@1@W4io_errc@1@@Z"(ptr dead_on_unwind noalias writable sret(%"class.std::error_code") align 8 %0, i32 noundef %1) local_unnamed_addr #9 comdat {
  %3 = load atomic i32, ptr @"?$TSS0@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@1@XZ@4HA" unordered, align 4
  %4 = load i32, ptr @_Init_thread_epoch, align 4
  %5 = icmp sgt i32 %3, %4
  br i1 %5, label %6, label %10, !prof !36

6:                                                ; preds = %2
  tail call void @_Init_thread_header(ptr nonnull @"?$TSS0@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@1@XZ@4HA") #10
  %7 = load atomic i32, ptr @"?$TSS0@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@1@XZ@4HA" unordered, align 4
  %8 = icmp eq i32 %7, -1
  br i1 %8, label %9, label %10

9:                                                ; preds = %6
  tail call void @_Init_thread_footer(ptr nonnull @"?$TSS0@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@1@XZ@4HA") #10
  br label %10

10:                                               ; preds = %2, %6, %9
  store i32 %1, ptr %0, align 8
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store ptr @"?_Static@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@1@XZ@4V21@A", ptr %11, align 8
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef ptr @"??0failure@ios_base@std@@QEAA@PEBDAEBVerror_code@2@@Z"(ptr noundef nonnull returned align 8 dereferenceable(40) %0, ptr noundef %1, ptr noundef nonnull align 8 dereferenceable(16) %2) unnamed_addr #2 comdat align 2 {
  %4 = alloca %"class.std::error_code", align 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %4, ptr noundef nonnull align 8 dereferenceable(16) %2, i64 16, i1 false)
  %5 = call noundef ptr @"??0system_error@std@@QEAA@Verror_code@1@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(40) %0, ptr dead_on_return noundef nonnull %4, ptr noundef %1)
  store ptr @"??_7failure@ios_base@std@@6B@", ptr %0, align 8
  ret ptr %0
}

; Function Attrs: inlinehint mustprogress nounwind uwtable
define linkonce_odr dso_local noundef ptr @"??0failure@ios_base@std@@QEAA@AEBV012@@Z"(ptr noundef nonnull returned align 8 dereferenceable(40) %0, ptr noundef nonnull align 8 dereferenceable(40) %1) unnamed_addr #9 comdat align 2 personality ptr @__CxxFrameHandler3 {
  store ptr @"??_7exception@std@@6B@", ptr %0, align 8
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %3, i8 0, i64 16, i1 false)
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  invoke void @__std_exception_copy(ptr noundef nonnull %4, ptr noundef nonnull %3)
          to label %7 unwind label %5

5:                                                ; preds = %2
  %6 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %6) ]
  unreachable

7:                                                ; preds = %2
  store ptr @"??_7_System_error@std@@6B@", ptr %0, align 8
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %9 = getelementptr inbounds nuw i8, ptr %1, i64 24
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %8, ptr noundef nonnull align 8 dereferenceable(16) %9, i64 16, i1 false)
  store ptr @"??_7failure@ios_base@std@@6B@", ptr %0, align 8
  ret ptr %0
}

; Function Attrs: inlinehint mustprogress nounwind uwtable
define linkonce_odr dso_local noundef ptr @"??0system_error@std@@QEAA@AEBV01@@Z"(ptr noundef nonnull returned align 8 dereferenceable(40) %0, ptr noundef nonnull align 8 dereferenceable(40) %1) unnamed_addr #9 comdat align 2 personality ptr @__CxxFrameHandler3 {
  store ptr @"??_7exception@std@@6B@", ptr %0, align 8
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %3, i8 0, i64 16, i1 false)
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  invoke void @__std_exception_copy(ptr noundef nonnull %4, ptr noundef nonnull %3)
          to label %7 unwind label %5

5:                                                ; preds = %2
  %6 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %6) ]
  unreachable

7:                                                ; preds = %2
  store ptr @"??_7_System_error@std@@6B@", ptr %0, align 8
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %9 = getelementptr inbounds nuw i8, ptr %1, i64 24
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %8, ptr noundef nonnull align 8 dereferenceable(16) %9, i64 16, i1 false)
  store ptr @"??_7system_error@std@@6B@", ptr %0, align 8
  ret ptr %0
}

; Function Attrs: inlinehint mustprogress nounwind uwtable
define linkonce_odr dso_local noundef ptr @"??0_System_error@std@@QEAA@AEBV01@@Z"(ptr noundef nonnull returned align 8 dereferenceable(40) %0, ptr noundef nonnull align 8 dereferenceable(40) %1) unnamed_addr #9 comdat align 2 personality ptr @__CxxFrameHandler3 {
  store ptr @"??_7exception@std@@6B@", ptr %0, align 8
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %3, i8 0, i64 16, i1 false)
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  invoke void @__std_exception_copy(ptr noundef nonnull %4, ptr noundef nonnull %3)
          to label %7 unwind label %5

5:                                                ; preds = %2
  %6 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %6) ]
  unreachable

7:                                                ; preds = %2
  store ptr @"??_7_System_error@std@@6B@", ptr %0, align 8
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %9 = getelementptr inbounds nuw i8, ptr %1, i64 24
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %8, ptr noundef nonnull align 8 dereferenceable(16) %9, i64 16, i1 false)
  ret ptr %0
}

; Function Attrs: inlinehint mustprogress nounwind uwtable
define linkonce_odr dso_local noundef ptr @"??0runtime_error@std@@QEAA@AEBV01@@Z"(ptr noundef nonnull returned align 8 dereferenceable(24) %0, ptr noundef nonnull align 8 dereferenceable(24) %1) unnamed_addr #9 comdat align 2 personality ptr @__CxxFrameHandler3 {
  store ptr @"??_7exception@std@@6B@", ptr %0, align 8
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %3, i8 0, i64 16, i1 false)
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  invoke void @__std_exception_copy(ptr noundef nonnull %4, ptr noundef nonnull %3)
          to label %7 unwind label %5

5:                                                ; preds = %2
  %6 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %6) ]
  unreachable

7:                                                ; preds = %2
  store ptr @"??_7runtime_error@std@@6B@", ptr %0, align 8
  ret ptr %0
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local noundef ptr @"??0exception@std@@QEAA@AEBV01@@Z"(ptr noundef nonnull returned align 8 dereferenceable(24) %0, ptr noundef nonnull align 8 dereferenceable(24) %1) unnamed_addr #7 comdat align 2 personality ptr @__CxxFrameHandler3 {
  store ptr @"??_7exception@std@@6B@", ptr %0, align 8
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %3, i8 0, i64 16, i1 false)
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  invoke void @__std_exception_copy(ptr noundef nonnull %4, ptr noundef nonnull %3)
          to label %5 unwind label %6

5:                                                ; preds = %2
  ret ptr %0

6:                                                ; preds = %2
  %7 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %7) ]
  unreachable
}

; Function Attrs: nounwind
declare dso_local void @_Init_thread_header(ptr) local_unnamed_addr #10

; Function Attrs: nounwind
declare dso_local void @_Init_thread_footer(ptr) local_unnamed_addr #10

; Function Attrs: inlinehint mustprogress nounwind uwtable
define linkonce_odr dso_local noundef ptr @"??_G_Iostream_error_category2@std@@UEAAPEAXI@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, i32 noundef %1) unnamed_addr #9 comdat align 2 {
  %3 = icmp eq i32 %1, 0
  br i1 %3, label %5, label %4

4:                                                ; preds = %2
  tail call void @"??3@YAXPEAX_K@Z"(ptr noundef %0, i64 noundef 16) #28
  br label %5

5:                                                ; preds = %4, %2
  ret ptr %0
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local noundef ptr @"?name@_Iostream_error_category2@std@@UEBAPEBDXZ"(ptr noundef nonnull align 8 dereferenceable(16) %0) unnamed_addr #7 comdat align 2 {
  ret ptr @"??_C@_08LLGCOLLL@iostream?$AA@"
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @"?message@_Iostream_error_category2@std@@UEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@H@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr dead_on_unwind noalias writable sret(%"class.std::basic_string") align 8 %1, i32 noundef %2) unnamed_addr #2 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %4 = icmp eq i32 %2, 1
  br i1 %4, label %5, label %10

5:                                                ; preds = %3
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %1, i8 0, i64 32, i1 false)
  %6 = tail call noalias noundef nonnull dereferenceable(32) ptr @"??2@YAPEAX_K@Z"(i64 noundef 32) #31
  store ptr %6, ptr %1, align 8
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store i64 21, ptr %7, align 8
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 24
  store i64 31, ptr %8, align 8
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(21) %6, ptr noundef nonnull align 16 dereferenceable(21) @"?_Iostream_error@?4??message@_Iostream_error_category2@std@@UEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@3@H@Z@4QBDB", i64 21, i1 false)
  %9 = getelementptr inbounds nuw i8, ptr %6, i64 21
  store i8 0, ptr %9, align 1
  br label %44

10:                                               ; preds = %3
  %11 = tail call noundef ptr @"?_Syserror_map@std@@YAPEBDH@Z"(i32 noundef %2)
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %1, i8 0, i64 32, i1 false)
  %12 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %11) #10
  %13 = icmp slt i64 %12, 0
  br i1 %13, label %14, label %15

14:                                               ; preds = %10
  tail call void @"?_Xlen_string@std@@YAXXZ"() #29
  unreachable

15:                                               ; preds = %10
  %16 = icmp ult i64 %12, 16
  br i1 %16, label %17, label %21

17:                                               ; preds = %15
  %18 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store i64 %12, ptr %18, align 8
  %19 = getelementptr inbounds nuw i8, ptr %1, i64 24
  store i64 15, ptr %19, align 8
  tail call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 8 dereferenceable(32) %1, ptr nonnull align 1 %11, i64 %12, i1 false)
  %20 = getelementptr inbounds nuw [16 x i8], ptr %1, i64 0, i64 %12
  store i8 0, ptr %20, align 1
  br label %44

21:                                               ; preds = %15
  %22 = or i64 %12, 15
  %23 = tail call i64 @llvm.umax.i64(i64 %22, i64 22)
  %24 = icmp ugt i64 %22, 4094
  br i1 %24, label %25, label %36

25:                                               ; preds = %21
  %26 = icmp ult i64 %22, -40
  br i1 %26, label %28, label %27

27:                                               ; preds = %25
  tail call void @"?_Throw_bad_array_new_length@std@@YAXXZ"() #29
  unreachable

28:                                               ; preds = %25
  %29 = add nuw i64 %23, 40
  %30 = tail call noalias noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef %29) #31
  %31 = ptrtoint ptr %30 to i64
  %32 = add i64 %31, 39
  %33 = and i64 %32, -32
  %34 = inttoptr i64 %33 to ptr
  %35 = getelementptr inbounds i8, ptr %34, i64 -8
  store i64 %31, ptr %35, align 8
  br label %39

36:                                               ; preds = %21
  %37 = add nuw nsw i64 %23, 1
  %38 = tail call noalias noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef %37) #31
  br label %39

39:                                               ; preds = %36, %28
  %40 = phi ptr [ %34, %28 ], [ %38, %36 ]
  store ptr %40, ptr %1, align 8
  %41 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store i64 %12, ptr %41, align 8
  %42 = getelementptr inbounds nuw i8, ptr %1, i64 24
  store i64 %23, ptr %42, align 8
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 1 %40, ptr nonnull align 1 %11, i64 %12, i1 false)
  %43 = getelementptr inbounds nuw i8, ptr %40, i64 %12
  store i8 0, ptr %43, align 1
  br label %44

44:                                               ; preds = %39, %17, %5
  ret void
}

; Function Attrs: inlinehint mustprogress nounwind uwtable
define linkonce_odr dso_local void @"?default_error_condition@error_category@std@@UEBA?AVerror_condition@2@H@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr dead_on_unwind noalias writable sret(%"class.std::error_condition") align 8 %1, i32 noundef %2) unnamed_addr #9 comdat align 2 {
  store i32 %2, ptr %1, align 8
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store ptr %0, ptr %4, align 8
  ret void
}

; Function Attrs: inlinehint mustprogress nounwind uwtable
define linkonce_odr dso_local noundef zeroext i1 @"?equivalent@error_category@std@@UEBA_NAEBVerror_code@2@H@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) unnamed_addr #9 comdat align 2 {
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %7 = load i64, ptr %6, align 8
  %8 = getelementptr inbounds nuw i8, ptr %5, i64 8
  %9 = load i64, ptr %8, align 8
  %10 = icmp eq i64 %7, %9
  %11 = load i32, ptr %1, align 8
  %12 = icmp eq i32 %11, %2
  %13 = select i1 %10, i1 %12, i1 false
  ret i1 %13
}

; Function Attrs: inlinehint mustprogress nounwind uwtable
define linkonce_odr dso_local noundef zeroext i1 @"?equivalent@error_category@std@@UEBA_NHAEBVerror_condition@2@@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, i32 noundef %1, ptr noundef nonnull align 8 dereferenceable(16) %2) unnamed_addr #9 comdat align 2 {
  %4 = alloca %"class.std::error_condition", align 8
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %4) #10
  %5 = load ptr, ptr %0, align 8
  %6 = getelementptr inbounds nuw i8, ptr %5, i64 24
  %7 = load ptr, ptr %6, align 8
  call void %7(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr dead_on_unwind nonnull writable sret(%"class.std::error_condition") align 8 %4, i32 noundef %1) #10
  %8 = getelementptr inbounds nuw i8, ptr %4, i64 8
  %9 = load ptr, ptr %8, align 8
  %10 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %11 = load ptr, ptr %10, align 8
  %12 = getelementptr inbounds nuw i8, ptr %9, i64 8
  %13 = load i64, ptr %12, align 8
  %14 = getelementptr inbounds nuw i8, ptr %11, i64 8
  %15 = load i64, ptr %14, align 8
  %16 = icmp eq i64 %13, %15
  %17 = load i32, ptr %4, align 8
  %18 = load i32, ptr %2, align 8
  %19 = icmp eq i32 %17, %18
  %20 = select i1 %16, i1 %19, i1 false
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %4) #10
  ret i1 %20
}

declare dso_local noundef ptr @"?_Syserror_map@std@@YAPEBDH@Z"(i32 noundef) local_unnamed_addr #11

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #12

; Function Attrs: inlinehint mustprogress noreturn uwtable
define linkonce_odr dso_local void @"?_Xlen_string@std@@YAXXZ"() local_unnamed_addr #13 comdat {
  tail call void @"?_Xlength_error@std@@YAXPEBD@Z"(ptr noundef nonnull @"??_C@_0BA@JFNIOLAK@string?5too?5long?$AA@") #29
  unreachable
}

; Function Attrs: noreturn
declare dso_local void @"?_Xlength_error@std@@YAXPEBD@Z"(ptr noundef) local_unnamed_addr #14

; Function Attrs: inlinehint mustprogress noreturn uwtable
define linkonce_odr dso_local void @"?_Throw_bad_array_new_length@std@@YAXXZ"() local_unnamed_addr #13 comdat {
  %1 = alloca %"class.std::bad_array_new_length", align 8
  store ptr @"??_7exception@std@@6B@", ptr %1, align 8
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 8
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %2, i8 0, i64 16, i1 false)
  store ptr @"??_C@_0BF@KINCDENJ@bad?5array?5new?5length?$AA@", ptr %2, align 8
  store ptr @"??_7bad_array_new_length@std@@6B@", ptr %1, align 8
  call void @_CxxThrowException(ptr nonnull %1, ptr nonnull @"_TI3?AVbad_array_new_length@std@@") #29
  unreachable
}

; Function Attrs: noreturn
declare dso_local void @_invalid_parameter_noinfo_noreturn() local_unnamed_addr #14

; Function Attrs: inlinehint mustprogress nounwind uwtable
define linkonce_odr dso_local noundef ptr @"??0bad_array_new_length@std@@QEAA@AEBV01@@Z"(ptr noundef nonnull returned align 8 dereferenceable(24) %0, ptr noundef nonnull align 8 dereferenceable(24) %1) unnamed_addr #9 comdat align 2 personality ptr @__CxxFrameHandler3 {
  store ptr @"??_7exception@std@@6B@", ptr %0, align 8
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %3, i8 0, i64 16, i1 false)
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  invoke void @__std_exception_copy(ptr noundef nonnull %4, ptr noundef nonnull %3)
          to label %7 unwind label %5

5:                                                ; preds = %2
  %6 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %6) ]
  unreachable

7:                                                ; preds = %2
  store ptr @"??_7bad_array_new_length@std@@6B@", ptr %0, align 8
  ret ptr %0
}

; Function Attrs: inlinehint mustprogress nounwind uwtable
define linkonce_odr dso_local noundef ptr @"??0bad_alloc@std@@QEAA@AEBV01@@Z"(ptr noundef nonnull returned align 8 dereferenceable(24) %0, ptr noundef nonnull align 8 dereferenceable(24) %1) unnamed_addr #9 comdat align 2 personality ptr @__CxxFrameHandler3 {
  store ptr @"??_7exception@std@@6B@", ptr %0, align 8
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %3, i8 0, i64 16, i1 false)
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  invoke void @__std_exception_copy(ptr noundef nonnull %4, ptr noundef nonnull %3)
          to label %7 unwind label %5

5:                                                ; preds = %2
  %6 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %6) ]
  unreachable

7:                                                ; preds = %2
  store ptr @"??_7bad_alloc@std@@6B@", ptr %0, align 8
  ret ptr %0
}

; Function Attrs: inlinehint mustprogress nounwind uwtable
define linkonce_odr dso_local noundef ptr @"??_Gbad_array_new_length@std@@UEAAPEAXI@Z"(ptr noundef nonnull align 8 dereferenceable(24) %0, i32 noundef %1) unnamed_addr #9 comdat align 2 personality ptr @__CxxFrameHandler3 {
  store ptr @"??_7exception@std@@6B@", ptr %0, align 8
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  invoke void @__std_exception_destroy(ptr noundef nonnull %3)
          to label %6 unwind label %4

4:                                                ; preds = %2
  %5 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %5) ]
  unreachable

6:                                                ; preds = %2
  %7 = icmp eq i32 %1, 0
  br i1 %7, label %9, label %8

8:                                                ; preds = %6
  tail call void @"??3@YAXPEAX_K@Z"(ptr noundef %0, i64 noundef 24) #28
  br label %9

9:                                                ; preds = %8, %6
  ret ptr %0
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local noundef ptr @"?what@exception@std@@UEBAPEBDXZ"(ptr noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #7 comdat align 2 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %3 = load ptr, ptr %2, align 8
  %4 = icmp eq ptr %3, null
  %5 = select i1 %4, ptr @"??_C@_0BC@EOODALEL@Unknown?5exception?$AA@", ptr %3
  ret ptr %5
}

; Function Attrs: inlinehint mustprogress nounwind uwtable
define linkonce_odr dso_local noundef ptr @"??_Gbad_alloc@std@@UEAAPEAXI@Z"(ptr noundef nonnull align 8 dereferenceable(24) %0, i32 noundef %1) unnamed_addr #9 comdat align 2 personality ptr @__CxxFrameHandler3 {
  store ptr @"??_7exception@std@@6B@", ptr %0, align 8
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  invoke void @__std_exception_destroy(ptr noundef nonnull %3)
          to label %6 unwind label %4

4:                                                ; preds = %2
  %5 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %5) ]
  unreachable

6:                                                ; preds = %2
  %7 = icmp eq i32 %1, 0
  br i1 %7, label %9, label %8

8:                                                ; preds = %6
  tail call void @"??3@YAXPEAX_K@Z"(ptr noundef %0, i64 noundef 24) #28
  br label %9

9:                                                ; preds = %8, %6
  ret ptr %0
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local noundef ptr @"??_Gexception@std@@UEAAPEAXI@Z"(ptr noundef nonnull align 8 dereferenceable(24) %0, i32 noundef %1) unnamed_addr #7 comdat align 2 personality ptr @__CxxFrameHandler3 {
  store ptr @"??_7exception@std@@6B@", ptr %0, align 8
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  invoke void @__std_exception_destroy(ptr noundef nonnull %3)
          to label %6 unwind label %4

4:                                                ; preds = %2
  %5 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %5) ]
  unreachable

6:                                                ; preds = %2
  %7 = icmp eq i32 %1, 0
  br i1 %7, label %9, label %8

8:                                                ; preds = %6
  tail call void @"??3@YAXPEAX_K@Z"(ptr noundef %0, i64 noundef 24) #28
  br label %9

9:                                                ; preds = %8, %6
  ret ptr %0
}

declare dso_local void @__std_exception_destroy(ptr noundef) local_unnamed_addr #11

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef ptr @"??0system_error@std@@QEAA@Verror_code@1@PEBD@Z"(ptr noundef nonnull returned align 8 dereferenceable(40) %0, ptr dead_on_return noundef %1, ptr noundef %2) unnamed_addr #2 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %4 = alloca %"class.std::basic_string", align 8
  %5 = alloca %"class.std::error_code", align 8
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %4) #10
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %4, i8 0, i64 32, i1 false)
  %6 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %2) #10
  %7 = icmp slt i64 %6, 0
  br i1 %7, label %8, label %9

8:                                                ; preds = %3
  tail call void @"?_Xlen_string@std@@YAXXZ"() #29
  unreachable

9:                                                ; preds = %3
  %10 = icmp ult i64 %6, 16
  br i1 %10, label %11, label %15

11:                                               ; preds = %9
  %12 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store i64 %6, ptr %12, align 8
  %13 = getelementptr inbounds nuw i8, ptr %4, i64 24
  store i64 15, ptr %13, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 8 dereferenceable(32) %4, ptr nonnull align 1 %2, i64 %6, i1 false)
  %14 = getelementptr inbounds nuw [16 x i8], ptr %4, i64 0, i64 %6
  store i8 0, ptr %14, align 1
  br label %38

15:                                               ; preds = %9
  %16 = or i64 %6, 15
  %17 = tail call i64 @llvm.umax.i64(i64 %16, i64 22)
  %18 = icmp ugt i64 %16, 4094
  br i1 %18, label %19, label %30

19:                                               ; preds = %15
  %20 = icmp ult i64 %16, -40
  br i1 %20, label %22, label %21

21:                                               ; preds = %19
  tail call void @"?_Throw_bad_array_new_length@std@@YAXXZ"() #29
  unreachable

22:                                               ; preds = %19
  %23 = add nuw i64 %17, 40
  %24 = tail call noalias noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef %23) #31
  %25 = ptrtoint ptr %24 to i64
  %26 = add i64 %25, 39
  %27 = and i64 %26, -32
  %28 = inttoptr i64 %27 to ptr
  %29 = getelementptr inbounds i8, ptr %28, i64 -8
  store i64 %25, ptr %29, align 8
  br label %33

30:                                               ; preds = %15
  %31 = add nuw nsw i64 %17, 1
  %32 = tail call noalias noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef %31) #31
  br label %33

33:                                               ; preds = %30, %22
  %34 = phi ptr [ %28, %22 ], [ %32, %30 ]
  store ptr %34, ptr %4, align 8
  %35 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store i64 %6, ptr %35, align 8
  %36 = getelementptr inbounds nuw i8, ptr %4, i64 24
  store i64 %17, ptr %36, align 8
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 1 %34, ptr nonnull align 1 %2, i64 %6, i1 false)
  %37 = getelementptr inbounds nuw i8, ptr %34, i64 %6
  store i8 0, ptr %37, align 1
  br label %38

38:                                               ; preds = %11, %33
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %5, ptr noundef nonnull align 8 dereferenceable(16) %1, i64 16, i1 false)
  %39 = invoke noundef ptr @"??0_System_error@std@@IEAA@Verror_code@1@AEBV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@1@@Z"(ptr noundef nonnull align 8 dereferenceable(40) %0, ptr dead_on_return noundef nonnull %5, ptr noundef nonnull align 8 dereferenceable(32) %4)
          to label %40 unwind label %67

40:                                               ; preds = %38
  %41 = getelementptr inbounds nuw i8, ptr %4, i64 24
  %42 = load i64, ptr %41, align 8
  %43 = icmp ugt i64 %42, 15
  br i1 %43, label %44, label %65

44:                                               ; preds = %40
  %45 = load ptr, ptr %4, align 8
  %46 = add i64 %42, 1
  %47 = icmp ugt i64 %46, 4095
  br i1 %47, label %48, label %62

48:                                               ; preds = %44
  %49 = getelementptr inbounds i8, ptr %45, i64 -8
  %50 = load i64, ptr %49, align 8
  %51 = ptrtoint ptr %45 to i64
  %52 = add i64 %51, -8
  %53 = sub i64 %52, %50
  %54 = icmp ult i64 %53, 32
  br i1 %54, label %57, label %55

55:                                               ; preds = %48
  invoke void @_invalid_parameter_noinfo_noreturn() #29
          to label %56 unwind label %60

56:                                               ; preds = %55
  unreachable

57:                                               ; preds = %48
  %58 = add i64 %42, 40
  %59 = inttoptr i64 %50 to ptr
  br label %62

60:                                               ; preds = %55
  %61 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %61) ]
  unreachable

62:                                               ; preds = %57, %44
  %63 = phi i64 [ %58, %57 ], [ %46, %44 ]
  %64 = phi ptr [ %59, %57 ], [ %45, %44 ]
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %64, i64 noundef %63) #10
  br label %65

65:                                               ; preds = %40, %62
  %66 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store i64 0, ptr %66, align 8
  store i64 15, ptr %41, align 8
  store i8 0, ptr %4, align 8
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %4) #10
  store ptr @"??_7system_error@std@@6B@", ptr %0, align 8
  ret ptr %0

67:                                               ; preds = %38
  %68 = cleanuppad within none []
  call void @"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %4) #10 [ "funclet"(token %68) ]
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %4) #10
  cleanupret from %68 unwind to caller
}

; Function Attrs: inlinehint mustprogress nounwind uwtable
define linkonce_odr dso_local noundef ptr @"??_Gfailure@ios_base@std@@UEAAPEAXI@Z"(ptr noundef nonnull align 8 dereferenceable(40) %0, i32 noundef %1) unnamed_addr #9 comdat align 2 personality ptr @__CxxFrameHandler3 {
  store ptr @"??_7exception@std@@6B@", ptr %0, align 8
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  invoke void @__std_exception_destroy(ptr noundef nonnull %3)
          to label %6 unwind label %4

4:                                                ; preds = %2
  %5 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %5) ]
  unreachable

6:                                                ; preds = %2
  %7 = icmp eq i32 %1, 0
  br i1 %7, label %9, label %8

8:                                                ; preds = %6
  tail call void @"??3@YAXPEAX_K@Z"(ptr noundef %0, i64 noundef 40) #28
  br label %9

9:                                                ; preds = %8, %6
  ret ptr %0
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef ptr @"??0_System_error@std@@IEAA@Verror_code@1@AEBV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@1@@Z"(ptr noundef nonnull returned align 8 dereferenceable(40) %0, ptr dead_on_return noundef %1, ptr noundef nonnull align 8 dereferenceable(32) %2) unnamed_addr #2 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %4 = alloca %struct.__std_exception_data, align 8
  %5 = alloca %"class.std::basic_string", align 8
  %6 = alloca %"class.std::basic_string", align 8
  %7 = alloca %"class.std::error_code", align 8
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %5) #10
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %6, i8 0, i64 32, i1 false)
  %8 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %9 = load i64, ptr %8, align 8
  %10 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %11 = load i64, ptr %10, align 8
  %12 = icmp ugt i64 %11, 15
  %13 = load ptr, ptr %2, align 8
  %14 = select i1 %12, ptr %13, ptr %2
  %15 = icmp slt i64 %9, 0
  br i1 %15, label %16, label %17

16:                                               ; preds = %3
  tail call void @"?_Xlen_string@std@@YAXXZ"() #29
  unreachable

17:                                               ; preds = %3
  %18 = icmp ult i64 %9, 16
  br i1 %18, label %19, label %22

19:                                               ; preds = %17
  %20 = getelementptr inbounds nuw i8, ptr %6, i64 16
  store i64 %9, ptr %20, align 8
  %21 = getelementptr inbounds nuw i8, ptr %6, i64 24
  store i64 15, ptr %21, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %6, ptr noundef nonnull align 1 dereferenceable(16) %14, i64 16, i1 false)
  br label %45

22:                                               ; preds = %17
  %23 = or i64 %9, 15
  %24 = tail call i64 @llvm.umax.i64(i64 %23, i64 22)
  %25 = icmp ugt i64 %23, 4094
  br i1 %25, label %26, label %37

26:                                               ; preds = %22
  %27 = icmp ult i64 %23, -40
  br i1 %27, label %29, label %28

28:                                               ; preds = %26
  tail call void @"?_Throw_bad_array_new_length@std@@YAXXZ"() #29
  unreachable

29:                                               ; preds = %26
  %30 = add nuw i64 %24, 40
  %31 = tail call noalias noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef %30) #31
  %32 = ptrtoint ptr %31 to i64
  %33 = add i64 %32, 39
  %34 = and i64 %33, -32
  %35 = inttoptr i64 %34 to ptr
  %36 = getelementptr inbounds i8, ptr %35, i64 -8
  store i64 %32, ptr %36, align 8
  br label %40

37:                                               ; preds = %22
  %38 = add nuw nsw i64 %24, 1
  %39 = tail call noalias noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef %38) #31
  br label %40

40:                                               ; preds = %37, %29
  %41 = phi ptr [ %35, %29 ], [ %39, %37 ]
  store ptr %41, ptr %6, align 8
  %42 = getelementptr inbounds nuw i8, ptr %6, i64 16
  store i64 %9, ptr %42, align 8
  %43 = getelementptr inbounds nuw i8, ptr %6, i64 24
  store i64 %24, ptr %43, align 8
  %44 = add nuw i64 %9, 1
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(1) %41, ptr noundef nonnull align 1 dereferenceable(1) %14, i64 %44, i1 false)
  br label %45

45:                                               ; preds = %19, %40
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %7, ptr noundef nonnull align 8 dereferenceable(16) %1, i64 16, i1 false)
  call void @"?_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z"(ptr dead_on_unwind nonnull writable sret(%"class.std::basic_string") align 8 %5, ptr dead_on_return noundef nonnull %7, ptr dead_on_return noundef nonnull %6)
  %46 = getelementptr inbounds nuw i8, ptr %5, i64 24
  %47 = load i64, ptr %46, align 8
  %48 = icmp ugt i64 %47, 15
  %49 = load ptr, ptr %5, align 8
  %50 = select i1 %48, ptr %49, ptr %5
  store ptr @"??_7exception@std@@6B@", ptr %0, align 8
  %51 = getelementptr inbounds nuw i8, ptr %0, i64 8
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %51, i8 0, i64 16, i1 false)
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %4) #10
  store ptr %50, ptr %4, align 8
  %52 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store i8 1, ptr %52, align 8
  invoke void @__std_exception_copy(ptr noundef nonnull %4, ptr noundef nonnull %51)
          to label %55 unwind label %53

53:                                               ; preds = %45
  %54 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %54) ]
  unreachable

55:                                               ; preds = %45
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %4) #10
  store ptr @"??_7runtime_error@std@@6B@", ptr %0, align 8
  %56 = load i64, ptr %46, align 8
  %57 = icmp ugt i64 %56, 15
  br i1 %57, label %58, label %79

58:                                               ; preds = %55
  %59 = load ptr, ptr %5, align 8
  %60 = add i64 %56, 1
  %61 = icmp ugt i64 %60, 4095
  br i1 %61, label %62, label %76

62:                                               ; preds = %58
  %63 = getelementptr inbounds i8, ptr %59, i64 -8
  %64 = load i64, ptr %63, align 8
  %65 = ptrtoint ptr %59 to i64
  %66 = add i64 %65, -8
  %67 = sub i64 %66, %64
  %68 = icmp ult i64 %67, 32
  br i1 %68, label %71, label %69

69:                                               ; preds = %62
  invoke void @_invalid_parameter_noinfo_noreturn() #29
          to label %70 unwind label %74

70:                                               ; preds = %69
  unreachable

71:                                               ; preds = %62
  %72 = add i64 %56, 40
  %73 = inttoptr i64 %64 to ptr
  br label %76

74:                                               ; preds = %69
  %75 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %75) ]
  unreachable

76:                                               ; preds = %71, %58
  %77 = phi i64 [ %72, %71 ], [ %60, %58 ]
  %78 = phi ptr [ %73, %71 ], [ %59, %58 ]
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %78, i64 noundef %77) #10
  br label %79

79:                                               ; preds = %55, %76
  %80 = getelementptr inbounds nuw i8, ptr %5, i64 16
  store i64 0, ptr %80, align 8
  store i64 15, ptr %46, align 8
  store i8 0, ptr %5, align 8
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %5) #10
  store ptr @"??_7_System_error@std@@6B@", ptr %0, align 8
  %81 = getelementptr inbounds nuw i8, ptr %0, i64 24
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %81, ptr noundef nonnull align 8 dereferenceable(16) %1, i64 16, i1 false)
  ret ptr %0
}

; Function Attrs: inlinehint mustprogress nounwind uwtable
define linkonce_odr dso_local void @"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %0) unnamed_addr #9 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %3 = load i64, ptr %2, align 8
  %4 = icmp ugt i64 %3, 15
  br i1 %4, label %5, label %26

5:                                                ; preds = %1
  %6 = load ptr, ptr %0, align 8
  %7 = add i64 %3, 1
  %8 = icmp ugt i64 %7, 4095
  br i1 %8, label %9, label %23

9:                                                ; preds = %5
  %10 = getelementptr inbounds i8, ptr %6, i64 -8
  %11 = load i64, ptr %10, align 8
  %12 = ptrtoint ptr %6 to i64
  %13 = add i64 %12, -8
  %14 = sub i64 %13, %11
  %15 = icmp ult i64 %14, 32
  br i1 %15, label %18, label %16

16:                                               ; preds = %9
  invoke void @_invalid_parameter_noinfo_noreturn() #29
          to label %17 unwind label %21

17:                                               ; preds = %16
  unreachable

18:                                               ; preds = %9
  %19 = add i64 %3, 40
  %20 = inttoptr i64 %11 to ptr
  br label %23

21:                                               ; preds = %16
  %22 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %22) ]
  unreachable

23:                                               ; preds = %18, %5
  %24 = phi i64 [ %19, %18 ], [ %7, %5 ]
  %25 = phi ptr [ %20, %18 ], [ %6, %5 ]
  tail call void @"??3@YAXPEAX_K@Z"(ptr noundef %25, i64 noundef %24) #10
  br label %26

26:                                               ; preds = %1, %23
  %27 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store i64 0, ptr %27, align 8
  store i64 15, ptr %2, align 8
  store i8 0, ptr %0, align 8
  ret void
}

; Function Attrs: inlinehint mustprogress nounwind uwtable
define linkonce_odr dso_local noundef ptr @"??_Gsystem_error@std@@UEAAPEAXI@Z"(ptr noundef nonnull align 8 dereferenceable(40) %0, i32 noundef %1) unnamed_addr #9 comdat align 2 personality ptr @__CxxFrameHandler3 {
  store ptr @"??_7exception@std@@6B@", ptr %0, align 8
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  invoke void @__std_exception_destroy(ptr noundef nonnull %3)
          to label %6 unwind label %4

4:                                                ; preds = %2
  %5 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %5) ]
  unreachable

6:                                                ; preds = %2
  %7 = icmp eq i32 %1, 0
  br i1 %7, label %9, label %8

8:                                                ; preds = %6
  tail call void @"??3@YAXPEAX_K@Z"(ptr noundef %0, i64 noundef 40) #28
  br label %9

9:                                                ; preds = %8, %6
  ret ptr %0
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @"?_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z"(ptr dead_on_unwind noalias writable sret(%"class.std::basic_string") align 8 %0, ptr dead_on_return noundef %1, ptr dead_on_return noundef %2) local_unnamed_addr #2 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %4 = alloca %"class.std::basic_string", align 8
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %6 = load i64, ptr %5, align 8
  %7 = icmp eq i64 %6, 0
  br i1 %7, label %22, label %8

8:                                                ; preds = %3
  %9 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %10 = load i64, ptr %9, align 8
  %11 = sub i64 %10, %6
  %12 = icmp ult i64 %11, 2
  br i1 %12, label %20, label %13

13:                                               ; preds = %8
  %14 = add i64 %6, 2
  store i64 %14, ptr %5, align 8
  %15 = icmp ugt i64 %10, 15
  %16 = load ptr, ptr %2, align 8
  %17 = select i1 %15, ptr %16, ptr %2
  %18 = getelementptr inbounds nuw i8, ptr %17, i64 %6
  store i16 8250, ptr %18, align 1
  %19 = getelementptr inbounds nuw i8, ptr %17, i64 %14
  store i8 0, ptr %19, align 1
  br label %22

20:                                               ; preds = %8
  %21 = invoke noundef nonnull align 8 dereferenceable(32) ptr @"??$_Reallocate_grow_by@V<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@QEBD_K@Z@PEBD_K@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??append@01@QEAAAEAV01@QEBD0@Z@PEBD_K@Z"(ptr noundef nonnull align 8 dereferenceable(32) %2, i64 noundef 2, i8 undef, ptr noundef nonnull @"??_C@_02LMMGGCAJ@?3?5?$AA@", i64 noundef 2)
          to label %22 unwind label %78

22:                                               ; preds = %13, %20, %3
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %4) #10
  %23 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %24 = load ptr, ptr %23, align 8, !noalias !37
  %25 = load i32, ptr %1, align 8, !noalias !37
  %26 = load ptr, ptr %24, align 8, !noalias !37
  %27 = getelementptr inbounds nuw i8, ptr %26, i64 16
  %28 = load ptr, ptr %27, align 8, !noalias !37
  invoke void %28(ptr noundef nonnull align 8 dereferenceable(16) %24, ptr dead_on_unwind nonnull writable sret(%"class.std::basic_string") align 8 %4, i32 noundef %25)
          to label %29 unwind label %78

29:                                               ; preds = %22
  %30 = getelementptr inbounds nuw i8, ptr %4, i64 16
  %31 = load i64, ptr %30, align 8
  %32 = getelementptr inbounds nuw i8, ptr %4, i64 24
  %33 = load i64, ptr %32, align 8
  %34 = icmp ugt i64 %33, 15
  %35 = load ptr, ptr %4, align 8
  %36 = select i1 %34, ptr %35, ptr %4
  %37 = load i64, ptr %5, align 8
  %38 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %39 = load i64, ptr %38, align 8
  %40 = sub i64 %39, %37
  %41 = icmp ugt i64 %31, %40
  br i1 %41, label %49, label %42

42:                                               ; preds = %29
  %43 = add i64 %37, %31
  store i64 %43, ptr %5, align 8
  %44 = icmp ugt i64 %39, 15
  %45 = load ptr, ptr %2, align 8
  %46 = select i1 %44, ptr %45, ptr %2
  %47 = getelementptr inbounds nuw i8, ptr %46, i64 %37
  call void @llvm.memmove.p0.p0.i64(ptr align 1 %47, ptr align 1 %36, i64 %31, i1 false)
  %48 = getelementptr inbounds nuw i8, ptr %46, i64 %43
  store i8 0, ptr %48, align 1
  br label %51

49:                                               ; preds = %29
  %50 = invoke noundef nonnull align 8 dereferenceable(32) ptr @"??$_Reallocate_grow_by@V<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@QEBD_K@Z@PEBD_K@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??append@01@QEAAAEAV01@QEBD0@Z@PEBD_K@Z"(ptr noundef nonnull align 8 dereferenceable(32) %2, i64 noundef %31, i8 undef, ptr noundef %36, i64 noundef %31)
          to label %51 unwind label %76

51:                                               ; preds = %42, %49
  %52 = load i64, ptr %32, align 8
  %53 = icmp ugt i64 %52, 15
  br i1 %53, label %54, label %75

54:                                               ; preds = %51
  %55 = load ptr, ptr %4, align 8
  %56 = add i64 %52, 1
  %57 = icmp ugt i64 %56, 4095
  br i1 %57, label %58, label %72

58:                                               ; preds = %54
  %59 = getelementptr inbounds i8, ptr %55, i64 -8
  %60 = load i64, ptr %59, align 8
  %61 = ptrtoint ptr %55 to i64
  %62 = add i64 %61, -8
  %63 = sub i64 %62, %60
  %64 = icmp ult i64 %63, 32
  br i1 %64, label %67, label %65

65:                                               ; preds = %58
  invoke void @_invalid_parameter_noinfo_noreturn() #29
          to label %66 unwind label %70

66:                                               ; preds = %65
  unreachable

67:                                               ; preds = %58
  %68 = add i64 %52, 40
  %69 = inttoptr i64 %60 to ptr
  br label %72

70:                                               ; preds = %65
  %71 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %71) ]
  unreachable

72:                                               ; preds = %67, %54
  %73 = phi i64 [ %68, %67 ], [ %56, %54 ]
  %74 = phi ptr [ %69, %67 ], [ %55, %54 ]
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %74, i64 noundef %73) #10
  br label %75

75:                                               ; preds = %51, %72
  store i64 0, ptr %30, align 8
  store i64 15, ptr %32, align 8
  store i8 0, ptr %4, align 8
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %4) #10
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false)
  store i64 0, ptr %5, align 8
  store i64 15, ptr %38, align 8
  store i8 0, ptr %2, align 1
  store i64 0, ptr %5, align 8
  store i64 15, ptr %38, align 8
  store i8 0, ptr %2, align 1
  ret void

76:                                               ; preds = %49
  %77 = cleanuppad within none []
  call void @"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %4) #10 [ "funclet"(token %77) ]
  cleanupret from %77 unwind label %78

78:                                               ; preds = %22, %20, %76
  %79 = cleanuppad within none []
  call void @"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %2) #10 [ "funclet"(token %79) ]
  cleanupret from %79 unwind to caller
}

; Function Attrs: inlinehint mustprogress nounwind uwtable
define linkonce_odr dso_local noundef ptr @"??_G_System_error@std@@UEAAPEAXI@Z"(ptr noundef nonnull align 8 dereferenceable(40) %0, i32 noundef %1) unnamed_addr #9 comdat align 2 personality ptr @__CxxFrameHandler3 {
  store ptr @"??_7exception@std@@6B@", ptr %0, align 8
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  invoke void @__std_exception_destroy(ptr noundef nonnull %3)
          to label %6 unwind label %4

4:                                                ; preds = %2
  %5 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %5) ]
  unreachable

6:                                                ; preds = %2
  %7 = icmp eq i32 %1, 0
  br i1 %7, label %9, label %8

8:                                                ; preds = %6
  tail call void @"??3@YAXPEAX_K@Z"(ptr noundef %0, i64 noundef 40) #28
  br label %9

9:                                                ; preds = %8, %6
  ret ptr %0
}

; Function Attrs: inlinehint mustprogress uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(32) ptr @"??$_Reallocate_grow_by@V<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@QEBD_K@Z@PEBD_K@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??append@01@QEAAAEAV01@QEBD0@Z@PEBD_K@Z"(ptr noundef nonnull align 8 dereferenceable(32) %0, i64 noundef %1, i8 %2, ptr noundef %3, i64 noundef %4) local_unnamed_addr #15 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %7 = load i64, ptr %6, align 8
  %8 = sub i64 9223372036854775807, %7
  %9 = icmp ult i64 %8, %1
  br i1 %9, label %10, label %11

10:                                               ; preds = %5
  tail call void @"?_Xlen_string@std@@YAXXZ"() #29
  unreachable

11:                                               ; preds = %5
  %12 = add i64 %7, %1
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %14 = load i64, ptr %13, align 8
  %15 = or i64 %12, 15
  %16 = icmp slt i64 %12, 0
  br i1 %16, label %24, label %17

17:                                               ; preds = %11
  %18 = lshr i64 %14, 1
  %19 = xor i64 %18, 9223372036854775807
  %20 = icmp ugt i64 %14, %19
  br i1 %20, label %24, label %21

21:                                               ; preds = %17
  %22 = add i64 %18, %14
  %23 = tail call i64 @llvm.umax.i64(i64 %15, i64 %22)
  br label %24

24:                                               ; preds = %11, %17, %21
  %25 = phi i64 [ %23, %21 ], [ 9223372036854775807, %11 ], [ 9223372036854775807, %17 ]
  %26 = add i64 %25, 1
  %27 = icmp eq i64 %26, 0
  br i1 %27, label %43, label %28

28:                                               ; preds = %24
  %29 = icmp ugt i64 %26, 4095
  br i1 %29, label %30, label %41

30:                                               ; preds = %28
  %31 = icmp ult i64 %26, -39
  br i1 %31, label %33, label %32

32:                                               ; preds = %30
  tail call void @"?_Throw_bad_array_new_length@std@@YAXXZ"() #29
  unreachable

33:                                               ; preds = %30
  %34 = add i64 %25, 40
  %35 = tail call noalias noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef %34) #31
  %36 = ptrtoint ptr %35 to i64
  %37 = add i64 %36, 39
  %38 = and i64 %37, -32
  %39 = inttoptr i64 %38 to ptr
  %40 = getelementptr inbounds i8, ptr %39, i64 -8
  store i64 %36, ptr %40, align 8
  br label %43

41:                                               ; preds = %28
  %42 = tail call noalias noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef %26) #31
  br label %43

43:                                               ; preds = %24, %33, %41
  %44 = phi ptr [ %39, %33 ], [ %42, %41 ], [ null, %24 ]
  store i64 %12, ptr %6, align 8
  store i64 %25, ptr %13, align 8
  %45 = icmp ugt i64 %14, 15
  br i1 %45, label %46, label %69

46:                                               ; preds = %43
  %47 = load ptr, ptr %0, align 8
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 1 %44, ptr align 1 %47, i64 %7, i1 false)
  %48 = getelementptr i8, ptr %44, i64 %7
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 1 %48, ptr align 1 %3, i64 %4, i1 false)
  %49 = getelementptr i8, ptr %48, i64 %4
  store i8 0, ptr %49, align 1
  %50 = add i64 %14, 1
  %51 = icmp ugt i64 %50, 4095
  br i1 %51, label %52, label %66

52:                                               ; preds = %46
  %53 = getelementptr inbounds i8, ptr %47, i64 -8
  %54 = load i64, ptr %53, align 8
  %55 = ptrtoint ptr %47 to i64
  %56 = add i64 %55, -8
  %57 = sub i64 %56, %54
  %58 = icmp ult i64 %57, 32
  br i1 %58, label %61, label %59

59:                                               ; preds = %52
  invoke void @_invalid_parameter_noinfo_noreturn() #29
          to label %60 unwind label %64

60:                                               ; preds = %59
  unreachable

61:                                               ; preds = %52
  %62 = add i64 %14, 40
  %63 = inttoptr i64 %54 to ptr
  br label %66

64:                                               ; preds = %59
  %65 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %65) ]
  unreachable

66:                                               ; preds = %46, %61
  %67 = phi i64 [ %62, %61 ], [ %50, %46 ]
  %68 = phi ptr [ %63, %61 ], [ %47, %46 ]
  tail call void @"??3@YAXPEAX_K@Z"(ptr noundef %68, i64 noundef %67) #10
  br label %72

69:                                               ; preds = %43
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 1 %44, ptr nonnull align 8 %0, i64 %7, i1 false)
  %70 = getelementptr i8, ptr %44, i64 %7
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 1 %70, ptr align 1 %3, i64 %4, i1 false)
  %71 = getelementptr i8, ptr %70, i64 %4
  store i8 0, ptr %71, align 1
  br label %72

72:                                               ; preds = %69, %66
  store ptr %44, ptr %0, align 8
  ret ptr %0
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memmove.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg) #3

; Function Attrs: inlinehint mustprogress nounwind uwtable
define linkonce_odr dso_local noundef ptr @"??_Gruntime_error@std@@UEAAPEAXI@Z"(ptr noundef nonnull align 8 dereferenceable(24) %0, i32 noundef %1) unnamed_addr #9 comdat align 2 personality ptr @__CxxFrameHandler3 {
  store ptr @"??_7exception@std@@6B@", ptr %0, align 8
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  invoke void @__std_exception_destroy(ptr noundef nonnull %3)
          to label %6 unwind label %4

4:                                                ; preds = %2
  %5 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %5) ]
  unreachable

6:                                                ; preds = %2
  %7 = icmp eq i32 %1, 0
  br i1 %7, label %9, label %8

8:                                                ; preds = %6
  tail call void @"??3@YAXPEAX_K@Z"(ptr noundef %0, i64 noundef 24) #28
  br label %9

9:                                                ; preds = %8, %6
  ret ptr %0
}

declare dso_local void @__std_exception_copy(ptr noundef, ptr noundef) local_unnamed_addr #11

; Function Attrs: nounwind
declare dso_local noundef zeroext i1 @"?uncaught_exception@std@@YA_NXZ"() local_unnamed_addr #16

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @"?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) local_unnamed_addr #7 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %2 = alloca %"class.std::error_code", align 8
  %3 = alloca %"class.std::ios_base::failure", align 8
  %4 = alloca %"class.std::error_code", align 8
  %5 = load ptr, ptr %0, align 8
  %6 = getelementptr inbounds nuw i8, ptr %5, i64 4
  %7 = load i32, ptr %6, align 4
  %8 = sext i32 %7 to i64
  %9 = getelementptr inbounds i8, ptr %0, i64 %8
  %10 = getelementptr inbounds nuw i8, ptr %9, i64 16
  %11 = load i32, ptr %10, align 8
  %12 = icmp eq i32 %11, 0
  br i1 %12, label %13, label %56

13:                                               ; preds = %1
  %14 = getelementptr inbounds nuw i8, ptr %9, i64 24
  %15 = load i32, ptr %14, align 8
  %16 = and i32 %15, 2
  %17 = icmp eq i32 %16, 0
  br i1 %17, label %56, label %18

18:                                               ; preds = %13
  %19 = getelementptr inbounds nuw i8, ptr %9, i64 72
  %20 = load ptr, ptr %19, align 8
  %21 = load ptr, ptr %20, align 8
  %22 = getelementptr inbounds nuw i8, ptr %21, i64 104
  %23 = load ptr, ptr %22, align 8
  %24 = invoke noundef i32 %23(ptr noundef nonnull align 8 dereferenceable(104) %20)
          to label %25 unwind label %52

25:                                               ; preds = %18
  %26 = icmp eq i32 %24, -1
  br i1 %26, label %27, label %56

27:                                               ; preds = %25
  %28 = load ptr, ptr %0, align 8
  %29 = getelementptr inbounds nuw i8, ptr %28, i64 4
  %30 = load i32, ptr %29, align 4
  %31 = sext i32 %30 to i64
  %32 = getelementptr inbounds i8, ptr %0, i64 %31
  %33 = getelementptr inbounds nuw i8, ptr %32, i64 16
  %34 = load i32, ptr %33, align 8
  call void @llvm.lifetime.start.p0(i64 40, ptr nonnull %3)
  %35 = and i32 %34, 19
  %36 = or disjoint i32 %35, 4
  store i32 %36, ptr %33, align 8
  %37 = getelementptr inbounds nuw i8, ptr %32, i64 20
  %38 = load i32, ptr %37, align 4
  %39 = and i32 %38, %36
  %40 = icmp eq i32 %39, 0
  br i1 %40, label %51, label %41

41:                                               ; preds = %27
  %42 = and i32 %38, 4
  %43 = icmp eq i32 %42, 0
  %44 = and i32 %39, 2
  %45 = icmp eq i32 %44, 0
  %46 = select i1 %45, ptr @"??_C@_0BF@OOHOMBOF@ios_base?3?3eofbit?5set?$AA@", ptr @"??_C@_0BG@FMKFHCIL@ios_base?3?3failbit?5set?$AA@"
  %47 = select i1 %43, ptr %46, ptr @"??_C@_0BF@PHHKMMFD@ios_base?3?3badbit?5set?$AA@"
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %4) #10
  call void @"?make_error_code@std@@YA?AVerror_code@1@W4io_errc@1@@Z"(ptr dead_on_unwind nonnull writable sret(%"class.std::error_code") align 8 %4, i32 noundef 1) #10
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %2)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %2, ptr noundef nonnull align 8 dereferenceable(16) %4, i64 16, i1 false)
  %48 = invoke noundef ptr @"??0system_error@std@@QEAA@Verror_code@1@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(40) %3, ptr dead_on_return noundef nonnull %2, ptr noundef nonnull %47)
          to label %49 unwind label %52

49:                                               ; preds = %41
  store ptr @"??_7failure@ios_base@std@@6B@", ptr %3, align 8
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %2)
  invoke void @_CxxThrowException(ptr nonnull %3, ptr nonnull @"_TI5?AVfailure@ios_base@std@@") #29
          to label %50 unwind label %52

50:                                               ; preds = %49
  unreachable

51:                                               ; preds = %27
  call void @llvm.lifetime.end.p0(i64 40, ptr nonnull %3)
  br label %56

52:                                               ; preds = %41, %49, %18
  %53 = catchswitch within none [label %54] unwind label %57

54:                                               ; preds = %52
  %55 = catchpad within %53 [ptr null, i32 64, ptr null]
  catchret from %55 to label %56

56:                                               ; preds = %51, %1, %13, %25, %54
  ret void

57:                                               ; preds = %52
  %58 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %58) ]
  unreachable
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(8) ptr @"?put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0, i8 noundef %1) local_unnamed_addr #2 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %3 = alloca %"class.std::error_code", align 8
  %4 = alloca %"class.std::ios_base::failure", align 8
  %5 = alloca %"class.std::error_code", align 8
  %6 = alloca %"class.std::basic_ostream<char>::sentry", align 8
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %6) #10
  store ptr %0, ptr %6, align 8
  %7 = load ptr, ptr %0, align 8
  %8 = getelementptr inbounds nuw i8, ptr %7, i64 4
  %9 = load i32, ptr %8, align 4
  %10 = sext i32 %9 to i64
  %11 = getelementptr inbounds i8, ptr %0, i64 %10
  %12 = getelementptr inbounds nuw i8, ptr %11, i64 72
  %13 = load ptr, ptr %12, align 8
  %14 = icmp eq ptr %13, null
  br i1 %14, label %19, label %15

15:                                               ; preds = %2
  %16 = load ptr, ptr %13, align 8
  %17 = getelementptr inbounds nuw i8, ptr %16, i64 8
  %18 = load ptr, ptr %17, align 8
  tail call void %18(ptr noundef nonnull align 8 dereferenceable(104) %13)
  br label %19

19:                                               ; preds = %15, %2
  %20 = load ptr, ptr %0, align 8
  %21 = getelementptr inbounds nuw i8, ptr %20, i64 4
  %22 = load i32, ptr %21, align 4
  %23 = sext i32 %22 to i64
  %24 = getelementptr inbounds i8, ptr %0, i64 %23
  %25 = getelementptr inbounds nuw i8, ptr %24, i64 16
  %26 = load i32, ptr %25, align 8
  %27 = icmp eq i32 %26, 0
  br i1 %27, label %30, label %28

28:                                               ; preds = %19
  %29 = getelementptr inbounds nuw i8, ptr %6, i64 8
  store i8 0, ptr %29, align 8
  br label %53

30:                                               ; preds = %19
  %31 = getelementptr inbounds nuw i8, ptr %24, i64 80
  %32 = load ptr, ptr %31, align 8
  %33 = icmp eq ptr %32, null
  %34 = icmp eq ptr %32, %0
  %35 = or i1 %33, %34
  br i1 %35, label %36, label %38

36:                                               ; preds = %30
  %37 = getelementptr inbounds nuw i8, ptr %6, i64 8
  store i8 1, ptr %37, align 8
  br label %53

38:                                               ; preds = %30
  %39 = invoke noundef nonnull align 8 dereferenceable(8) ptr @"?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %32)
          to label %40 unwind label %51

40:                                               ; preds = %38
  %41 = load ptr, ptr %0, align 8
  %42 = getelementptr inbounds nuw i8, ptr %41, i64 4
  %43 = load i32, ptr %42, align 4
  %44 = sext i32 %43 to i64
  %45 = getelementptr inbounds i8, ptr %0, i64 %44
  %46 = getelementptr inbounds nuw i8, ptr %45, i64 16
  %47 = load i32, ptr %46, align 8
  %48 = icmp eq i32 %47, 0
  %49 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %50 = zext i1 %48 to i8
  store i8 %50, ptr %49, align 8
  br label %53

51:                                               ; preds = %38
  %52 = cleanuppad within none []
  call void @"??1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %6) #10 [ "funclet"(token %52) ]
  cleanupret from %52 unwind to caller

53:                                               ; preds = %28, %36, %40
  %54 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %55 = load i8, ptr %54, align 8, !range !18, !noundef !19
  %56 = trunc nuw i8 %55 to i1
  br i1 %56, label %57, label %100

57:                                               ; preds = %53
  %58 = load ptr, ptr %0, align 8
  %59 = getelementptr inbounds nuw i8, ptr %58, i64 4
  %60 = load i32, ptr %59, align 4
  %61 = sext i32 %60 to i64
  %62 = getelementptr inbounds i8, ptr %0, i64 %61
  %63 = getelementptr inbounds nuw i8, ptr %62, i64 72
  %64 = load ptr, ptr %63, align 8
  %65 = getelementptr inbounds nuw i8, ptr %64, i64 64
  %66 = load ptr, ptr %65, align 8
  %67 = load ptr, ptr %66, align 8
  %68 = icmp eq ptr %67, null
  br i1 %68, label %80, label %69

69:                                               ; preds = %57
  %70 = getelementptr inbounds nuw i8, ptr %64, i64 88
  %71 = load ptr, ptr %70, align 8
  %72 = load i32, ptr %71, align 4
  %73 = icmp sgt i32 %72, 0
  br i1 %73, label %74, label %80

74:                                               ; preds = %69
  %75 = add nsw i32 %72, -1
  store i32 %75, ptr %71, align 4
  %76 = load ptr, ptr %65, align 8
  %77 = load ptr, ptr %76, align 8
  %78 = getelementptr inbounds nuw i8, ptr %77, i64 1
  store ptr %78, ptr %76, align 8
  store i8 %1, ptr %77, align 1
  %79 = zext i8 %1 to i32
  br label %96

80:                                               ; preds = %69, %57
  %81 = zext i8 %1 to i32
  %82 = load ptr, ptr %64, align 8
  %83 = getelementptr inbounds nuw i8, ptr %82, i64 24
  %84 = load ptr, ptr %83, align 8
  %85 = invoke noundef i32 %84(ptr noundef nonnull align 8 dereferenceable(104) %64, i32 noundef %81)
          to label %96 unwind label %86

86:                                               ; preds = %80
  %87 = catchswitch within none [label %88] unwind label %151

88:                                               ; preds = %86
  %89 = catchpad within %87 [ptr null, i32 64, ptr null]
  %90 = load ptr, ptr %0, align 8
  %91 = getelementptr inbounds nuw i8, ptr %90, i64 4
  %92 = load i32, ptr %91, align 4
  %93 = sext i32 %92 to i64
  %94 = getelementptr inbounds i8, ptr %0, i64 %93
  invoke void @"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"(ptr noundef nonnull align 8 dereferenceable(96) %94, i32 noundef 4, i1 noundef zeroext true) [ "funclet"(token %89) ]
          to label %95 unwind label %151

95:                                               ; preds = %88
  catchret from %89 to label %100

96:                                               ; preds = %74, %80
  %97 = phi i32 [ %79, %74 ], [ %85, %80 ]
  %98 = icmp eq i32 %97, -1
  %99 = select i1 %98, i32 4, i32 0
  br label %100

100:                                              ; preds = %96, %53, %95
  %101 = phi i32 [ 0, %95 ], [ 4, %53 ], [ %99, %96 ]
  %102 = load ptr, ptr %0, align 8
  %103 = getelementptr inbounds nuw i8, ptr %102, i64 4
  %104 = load i32, ptr %103, align 4
  %105 = sext i32 %104 to i64
  %106 = getelementptr inbounds i8, ptr %0, i64 %105
  %107 = getelementptr inbounds nuw i8, ptr %106, i64 16
  %108 = load i32, ptr %107, align 8
  %109 = getelementptr inbounds nuw i8, ptr %106, i64 72
  %110 = load ptr, ptr %109, align 8
  %111 = icmp eq ptr %110, null
  %112 = select i1 %111, i32 4, i32 0
  call void @llvm.lifetime.start.p0(i64 40, ptr nonnull %4)
  %113 = and i32 %108, 23
  %114 = or i32 %113, %101
  %115 = or i32 %114, %112
  store i32 %115, ptr %107, align 8
  %116 = getelementptr inbounds nuw i8, ptr %106, i64 20
  %117 = load i32, ptr %116, align 4
  %118 = and i32 %117, %115
  %119 = icmp eq i32 %118, 0
  br i1 %119, label %130, label %120

120:                                              ; preds = %100
  %121 = and i32 %118, 4
  %122 = icmp eq i32 %121, 0
  %123 = and i32 %118, 2
  %124 = icmp eq i32 %123, 0
  %125 = select i1 %124, ptr @"??_C@_0BF@OOHOMBOF@ios_base?3?3eofbit?5set?$AA@", ptr @"??_C@_0BG@FMKFHCIL@ios_base?3?3failbit?5set?$AA@"
  %126 = select i1 %122, ptr %125, ptr @"??_C@_0BF@PHHKMMFD@ios_base?3?3badbit?5set?$AA@"
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %5) #10
  call void @"?make_error_code@std@@YA?AVerror_code@1@W4io_errc@1@@Z"(ptr dead_on_unwind nonnull writable sret(%"class.std::error_code") align 8 %5, i32 noundef 1) #10
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %3)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %3, ptr noundef nonnull align 8 dereferenceable(16) %5, i64 16, i1 false)
  %127 = invoke noundef ptr @"??0system_error@std@@QEAA@Verror_code@1@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(40) %4, ptr dead_on_return noundef nonnull %3, ptr noundef nonnull %126)
          to label %128 unwind label %151

128:                                              ; preds = %120
  store ptr @"??_7failure@ios_base@std@@6B@", ptr %4, align 8
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %3)
  invoke void @_CxxThrowException(ptr nonnull %4, ptr nonnull @"_TI5?AVfailure@ios_base@std@@") #29
          to label %129 unwind label %151

129:                                              ; preds = %128
  unreachable

130:                                              ; preds = %100
  call void @llvm.lifetime.end.p0(i64 40, ptr nonnull %4)
  %131 = tail call noundef zeroext i1 @"?uncaught_exception@std@@YA_NXZ"() #10
  br i1 %131, label %134, label %132

132:                                              ; preds = %130
  %133 = load ptr, ptr %6, align 8, !nonnull !19, !align !20
  tail call void @"?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ"(ptr noundef nonnull align 8 dereferenceable(8) %133) #10
  br label %134

134:                                              ; preds = %132, %130
  %135 = load ptr, ptr %6, align 8, !nonnull !19, !align !20
  %136 = load ptr, ptr %135, align 8
  %137 = getelementptr inbounds nuw i8, ptr %136, i64 4
  %138 = load i32, ptr %137, align 4
  %139 = sext i32 %138 to i64
  %140 = getelementptr inbounds i8, ptr %135, i64 %139
  %141 = getelementptr inbounds nuw i8, ptr %140, i64 72
  %142 = load ptr, ptr %141, align 8
  %143 = icmp eq ptr %142, null
  br i1 %143, label %150, label %144

144:                                              ; preds = %134
  %145 = load ptr, ptr %142, align 8
  %146 = getelementptr inbounds nuw i8, ptr %145, i64 16
  %147 = load ptr, ptr %146, align 8
  invoke void %147(ptr noundef nonnull align 8 dereferenceable(104) %142)
          to label %150 unwind label %148

148:                                              ; preds = %144
  %149 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %149) ]
  unreachable

150:                                              ; preds = %134, %144
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %6) #10
  ret ptr %0

151:                                              ; preds = %120, %128, %88, %86
  %152 = cleanuppad within none []
  call void @"??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %6) #10 [ "funclet"(token %152) ]
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %6) #10
  cleanupret from %152 unwind to caller
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(48) ptr @"??$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0) local_unnamed_addr #2 comdat personality ptr @__CxxFrameHandler3 {
  %2 = alloca %"class.std::_Locinfo", align 8
  %3 = alloca %"class.std::_Lockit", align 4
  %4 = alloca %"class.std::_Lockit", align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %4) #10
  %5 = call noundef ptr @"??0_Lockit@std@@QEAA@H@Z"(ptr noundef nonnull align 4 dereferenceable(4) %4, i32 noundef 0) #10
  %6 = load ptr, ptr @"?_Psave@?$_Facetptr@V?$ctype@D@std@@@std@@2PEBVfacet@locale@2@EB", align 8
  %7 = load i64, ptr @"?id@?$ctype@D@std@@2V0locale@2@A", align 8
  %8 = icmp eq i64 %7, 0
  br i1 %8, label %9, label %18

9:                                                ; preds = %1
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %3) #10
  %10 = call noundef ptr @"??0_Lockit@std@@QEAA@H@Z"(ptr noundef nonnull align 4 dereferenceable(4) %3, i32 noundef 0) #10
  %11 = load i64, ptr @"?id@?$ctype@D@std@@2V0locale@2@A", align 8
  %12 = icmp eq i64 %11, 0
  br i1 %12, label %13, label %17

13:                                               ; preds = %9
  %14 = load i32, ptr @"?_Id_cnt@id@locale@std@@0HA", align 4
  %15 = add nsw i32 %14, 1
  store i32 %15, ptr @"?_Id_cnt@id@locale@std@@0HA", align 4
  %16 = sext i32 %15 to i64
  store i64 %16, ptr @"?id@?$ctype@D@std@@2V0locale@2@A", align 8
  br label %17

17:                                               ; preds = %13, %9
  call void @"??1_Lockit@std@@QEAA@XZ"(ptr noundef nonnull align 4 dereferenceable(4) %3) #10
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %3) #10
  br label %18

18:                                               ; preds = %1, %17
  %19 = load i64, ptr @"?id@?$ctype@D@std@@2V0locale@2@A", align 8
  %20 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %21 = load ptr, ptr %20, align 8
  %22 = getelementptr inbounds nuw i8, ptr %21, i64 24
  %23 = load i64, ptr %22, align 8
  %24 = icmp ult i64 %19, %23
  br i1 %24, label %25, label %30

25:                                               ; preds = %18
  %26 = getelementptr inbounds nuw i8, ptr %21, i64 16
  %27 = load ptr, ptr %26, align 8
  %28 = getelementptr inbounds nuw ptr, ptr %27, i64 %19
  %29 = load ptr, ptr %28, align 8
  br label %30

30:                                               ; preds = %25, %18
  %31 = phi ptr [ %29, %25 ], [ null, %18 ]
  %32 = icmp eq ptr %31, null
  br i1 %32, label %33, label %48

33:                                               ; preds = %30
  %34 = getelementptr inbounds nuw i8, ptr %21, i64 36
  %35 = load i8, ptr %34, align 4, !range !18, !noundef !19
  %36 = trunc nuw i8 %35 to i1
  br i1 %36, label %37, label %48

37:                                               ; preds = %33
  %38 = invoke noundef ptr @"?_Getgloballocale@locale@std@@CAPEAV_Locimp@12@XZ"()
          to label %39 unwind label %83

39:                                               ; preds = %37
  %40 = getelementptr inbounds nuw i8, ptr %38, i64 24
  %41 = load i64, ptr %40, align 8
  %42 = icmp ult i64 %19, %41
  br i1 %42, label %43, label %48

43:                                               ; preds = %39
  %44 = getelementptr inbounds nuw i8, ptr %38, i64 16
  %45 = load ptr, ptr %44, align 8
  %46 = getelementptr inbounds nuw ptr, ptr %45, i64 %19
  %47 = load ptr, ptr %46, align 8
  br label %48

48:                                               ; preds = %43, %39, %33, %30
  %49 = phi ptr [ %31, %33 ], [ %31, %30 ], [ %47, %43 ], [ null, %39 ]
  %50 = icmp eq ptr %49, null
  br i1 %50, label %51, label %81

51:                                               ; preds = %48
  %52 = icmp eq ptr %6, null
  br i1 %52, label %53, label %81

53:                                               ; preds = %51
  %54 = invoke noalias noundef nonnull dereferenceable(48) ptr @"??2@YAPEAX_K@Z"(i64 noundef 48) #27
          to label %55 unwind label %83

55:                                               ; preds = %53
  call void @llvm.lifetime.start.p0(i64 104, ptr nonnull %2) #10
  %56 = load ptr, ptr %20, align 8
  %57 = icmp eq ptr %56, null
  br i1 %57, label %64, label %58

58:                                               ; preds = %55
  %59 = getelementptr inbounds nuw i8, ptr %56, i64 40
  %60 = load ptr, ptr %59, align 8
  %61 = icmp eq ptr %60, null
  %62 = getelementptr inbounds nuw i8, ptr %56, i64 48
  %63 = select i1 %61, ptr %62, ptr %60
  br label %64

64:                                               ; preds = %58, %55
  %65 = phi ptr [ %63, %58 ], [ @"??_C@_00CNPNBAHC@?$AA@", %55 ]
  %66 = invoke noundef ptr @"??0_Locinfo@std@@QEAA@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(104) %2, ptr noundef %65)
          to label %67 unwind label %70

67:                                               ; preds = %64
  store ptr @"??_7facet@locale@std@@6B@", ptr %54, align 8
  %68 = getelementptr inbounds nuw i8, ptr %54, i64 8
  store i32 0, ptr %68, align 8
  store ptr @"??_7?$ctype@D@std@@6B@", ptr %54, align 8
  %69 = getelementptr inbounds nuw i8, ptr %54, i64 16
  call void @_Getctype(ptr dead_on_unwind nonnull writable sret(%struct._Ctypevec) align 8 %69) #10
  call void @"??1_Locinfo@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(104) %2) #10
  call void @llvm.lifetime.end.p0(i64 104, ptr nonnull %2) #10
  invoke void @"?_Facet_Register@std@@YAXPEAV_Facet_base@1@@Z"(ptr noundef nonnull %54)
          to label %72 unwind label %76

70:                                               ; preds = %64
  %71 = cleanuppad within none []
  call void @llvm.lifetime.end.p0(i64 104, ptr nonnull %2) #10
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %54, i64 noundef 48) #28 [ "funclet"(token %71) ]
  cleanupret from %71 unwind label %83

72:                                               ; preds = %67
  %73 = load ptr, ptr %54, align 8
  %74 = getelementptr inbounds nuw i8, ptr %73, i64 8
  %75 = load ptr, ptr %74, align 8
  call void %75(ptr noundef nonnull align 8 dereferenceable(16) %54) #10
  store ptr %54, ptr @"?_Psave@?$_Facetptr@V?$ctype@D@std@@@std@@2PEBVfacet@locale@2@EB", align 8
  br label %81

76:                                               ; preds = %67
  %77 = cleanuppad within none []
  %78 = load ptr, ptr %54, align 8
  %79 = load ptr, ptr %78, align 8
  %80 = call noundef ptr %79(ptr noundef nonnull align 8 dereferenceable(8) %54, i32 noundef 1) #10 [ "funclet"(token %77) ]
  cleanupret from %77 unwind label %83

81:                                               ; preds = %51, %72, %48
  %82 = phi ptr [ %49, %48 ], [ %54, %72 ], [ %6, %51 ]
  call void @"??1_Lockit@std@@QEAA@XZ"(ptr noundef nonnull align 4 dereferenceable(4) %4) #10
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %4) #10
  ret ptr %82

83:                                               ; preds = %53, %70, %37, %76
  %84 = cleanuppad within none []
  call void @"??1_Lockit@std@@QEAA@XZ"(ptr noundef nonnull align 4 dereferenceable(4) %4) #10 [ "funclet"(token %84) ]
  cleanupret from %84 unwind to caller
}

; Function Attrs: nounwind
declare dso_local noundef ptr @"??0_Lockit@std@@QEAA@H@Z"(ptr noundef nonnull returned align 4 dereferenceable(4), i32 noundef) unnamed_addr #16

; Function Attrs: inlinehint mustprogress noreturn uwtable
define linkonce_odr dso_local void @"?_Throw_bad_cast@std@@YAXXZ"() local_unnamed_addr #13 comdat {
  %1 = alloca %"class.std::bad_cast", align 8
  store ptr @"??_7exception@std@@6B@", ptr %1, align 8
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 8
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %2, i8 0, i64 16, i1 false)
  store ptr @"??_C@_08EPJLHIJG@bad?5cast?$AA@", ptr %2, align 8
  store ptr @"??_7bad_cast@std@@6B@", ptr %1, align 8
  call void @_CxxThrowException(ptr nonnull %1, ptr nonnull @"_TI2?AVbad_cast@std@@") #29
  unreachable
}

declare dso_local void @"?_Facet_Register@std@@YAXPEAV_Facet_base@1@@Z"(ptr noundef) local_unnamed_addr #11

; Function Attrs: nounwind
declare dso_local void @"??1_Lockit@std@@QEAA@XZ"(ptr noundef nonnull align 4 dereferenceable(4)) unnamed_addr #16

declare dso_local noundef ptr @"?_Getgloballocale@locale@std@@CAPEAV_Locimp@12@XZ"() local_unnamed_addr #11

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef ptr @"??0_Locinfo@std@@QEAA@PEBD@Z"(ptr noundef nonnull returned align 8 dereferenceable(104) %0, ptr noundef %1) unnamed_addr #2 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %3 = tail call noundef ptr @"??0_Lockit@std@@QEAA@H@Z"(ptr noundef nonnull align 4 dereferenceable(4) %0, i32 noundef 0) #10
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store ptr null, ptr %4, align 8
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store i8 0, ptr %5, align 8
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 24
  store ptr null, ptr %6, align 8
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 32
  store i8 0, ptr %7, align 8
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 40
  store ptr null, ptr %8, align 8
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 48
  store i16 0, ptr %9, align 8
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 56
  store ptr null, ptr %10, align 8
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 64
  store i16 0, ptr %11, align 8
  %12 = getelementptr inbounds nuw i8, ptr %0, i64 72
  store ptr null, ptr %12, align 8
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 80
  store i8 0, ptr %13, align 8
  %14 = getelementptr inbounds nuw i8, ptr %0, i64 88
  store ptr null, ptr %14, align 8
  %15 = getelementptr inbounds nuw i8, ptr %0, i64 96
  store i8 0, ptr %15, align 8
  %16 = icmp eq ptr %1, null
  br i1 %16, label %19, label %17

17:                                               ; preds = %2
  invoke void @"?_Locinfo_ctor@_Locinfo@std@@SAXPEAV12@PEBD@Z"(ptr noundef nonnull %0, ptr noundef nonnull %1)
          to label %18 unwind label %21

18:                                               ; preds = %17
  ret ptr %0

19:                                               ; preds = %2
  invoke void @"?_Xruntime_error@std@@YAXPEBD@Z"(ptr noundef nonnull @"??_C@_0BA@ELKIONDK@bad?5locale?5name?$AA@") #29
          to label %20 unwind label %21

20:                                               ; preds = %19
  unreachable

21:                                               ; preds = %19, %17
  %22 = cleanuppad within none []
  %23 = load ptr, ptr %14, align 8
  %24 = icmp eq ptr %23, null
  br i1 %24, label %26, label %25

25:                                               ; preds = %21
  call void @free(ptr noundef %23) [ "funclet"(token %22) ]
  br label %26

26:                                               ; preds = %21, %25
  store ptr null, ptr %14, align 8
  %27 = load ptr, ptr %12, align 8
  %28 = icmp eq ptr %27, null
  br i1 %28, label %30, label %29

29:                                               ; preds = %26
  call void @free(ptr noundef %27) [ "funclet"(token %22) ]
  br label %30

30:                                               ; preds = %26, %29
  store ptr null, ptr %12, align 8
  %31 = load ptr, ptr %10, align 8
  %32 = icmp eq ptr %31, null
  br i1 %32, label %34, label %33

33:                                               ; preds = %30
  call void @free(ptr noundef %31) [ "funclet"(token %22) ]
  br label %34

34:                                               ; preds = %30, %33
  store ptr null, ptr %10, align 8
  %35 = load ptr, ptr %8, align 8
  %36 = icmp eq ptr %35, null
  br i1 %36, label %38, label %37

37:                                               ; preds = %34
  call void @free(ptr noundef %35) [ "funclet"(token %22) ]
  br label %38

38:                                               ; preds = %34, %37
  store ptr null, ptr %8, align 8
  %39 = load ptr, ptr %6, align 8
  %40 = icmp eq ptr %39, null
  br i1 %40, label %42, label %41

41:                                               ; preds = %38
  call void @free(ptr noundef %39) [ "funclet"(token %22) ]
  br label %42

42:                                               ; preds = %38, %41
  store ptr null, ptr %6, align 8
  %43 = load ptr, ptr %4, align 8
  %44 = icmp eq ptr %43, null
  br i1 %44, label %46, label %45

45:                                               ; preds = %42
  call void @free(ptr noundef %43) [ "funclet"(token %22) ]
  br label %46

46:                                               ; preds = %42, %45
  store ptr null, ptr %4, align 8
  call void @"??1_Lockit@std@@QEAA@XZ"(ptr noundef nonnull align 4 dereferenceable(4) %0) #10 [ "funclet"(token %22) ]
  cleanupret from %22 unwind to caller
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @"??1_Locinfo@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(104) %0) unnamed_addr #7 comdat align 2 personality ptr @__CxxFrameHandler3 {
  invoke void @"?_Locinfo_dtor@_Locinfo@std@@SAXPEAV12@@Z"(ptr noundef nonnull %0)
          to label %2 unwind label %33

2:                                                ; preds = %1
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 88
  %4 = load ptr, ptr %3, align 8
  %5 = icmp eq ptr %4, null
  br i1 %5, label %7, label %6

6:                                                ; preds = %2
  tail call void @free(ptr noundef %4)
  br label %7

7:                                                ; preds = %2, %6
  store ptr null, ptr %3, align 8
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 72
  %9 = load ptr, ptr %8, align 8
  %10 = icmp eq ptr %9, null
  br i1 %10, label %12, label %11

11:                                               ; preds = %7
  tail call void @free(ptr noundef %9)
  br label %12

12:                                               ; preds = %7, %11
  store ptr null, ptr %8, align 8
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 56
  %14 = load ptr, ptr %13, align 8
  %15 = icmp eq ptr %14, null
  br i1 %15, label %17, label %16

16:                                               ; preds = %12
  tail call void @free(ptr noundef %14)
  br label %17

17:                                               ; preds = %12, %16
  store ptr null, ptr %13, align 8
  %18 = getelementptr inbounds nuw i8, ptr %0, i64 40
  %19 = load ptr, ptr %18, align 8
  %20 = icmp eq ptr %19, null
  br i1 %20, label %22, label %21

21:                                               ; preds = %17
  tail call void @free(ptr noundef %19)
  br label %22

22:                                               ; preds = %17, %21
  store ptr null, ptr %18, align 8
  %23 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %24 = load ptr, ptr %23, align 8
  %25 = icmp eq ptr %24, null
  br i1 %25, label %27, label %26

26:                                               ; preds = %22
  tail call void @free(ptr noundef %24)
  br label %27

27:                                               ; preds = %22, %26
  store ptr null, ptr %23, align 8
  %28 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %29 = load ptr, ptr %28, align 8
  %30 = icmp eq ptr %29, null
  br i1 %30, label %32, label %31

31:                                               ; preds = %27
  tail call void @free(ptr noundef %29)
  br label %32

32:                                               ; preds = %27, %31
  store ptr null, ptr %28, align 8
  tail call void @"??1_Lockit@std@@QEAA@XZ"(ptr noundef nonnull align 4 dereferenceable(4) %0) #10
  ret void

33:                                               ; preds = %1
  %34 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %34) ]
  unreachable
}

declare dso_local void @"?_Locinfo_ctor@_Locinfo@std@@SAXPEAV12@PEBD@Z"(ptr noundef, ptr noundef) local_unnamed_addr #11

; Function Attrs: noreturn
declare dso_local void @"?_Xruntime_error@std@@YAXPEBD@Z"(ptr noundef) local_unnamed_addr #14

; Function Attrs: mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite)
declare dso_local void @free(ptr allocptr noundef captures(none)) local_unnamed_addr #17

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local noundef ptr @"??_G?$ctype@D@std@@MEAAPEAXI@Z"(ptr noundef nonnull align 8 dereferenceable(48) %0, i32 noundef %1) unnamed_addr #7 comdat align 2 personality ptr @__CxxFrameHandler3 {
  store ptr @"??_7?$ctype@D@std@@6B@", ptr %0, align 8
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %4 = load i32, ptr %3, align 8
  %5 = icmp sgt i32 %4, 0
  br i1 %5, label %6, label %9

6:                                                ; preds = %2
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %8 = load ptr, ptr %7, align 8
  tail call void @free(ptr noundef %8)
  br label %16

9:                                                ; preds = %2
  %10 = icmp slt i32 %4, 0
  br i1 %10, label %11, label %16

11:                                               ; preds = %9
  %12 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %13 = load ptr, ptr %12, align 8
  %14 = icmp eq ptr %13, null
  br i1 %14, label %16, label %15

15:                                               ; preds = %11
  tail call void @"??_V@YAXPEAX@Z"(ptr noundef %13) #28
  br label %16

16:                                               ; preds = %6, %9, %11, %15
  %17 = getelementptr inbounds nuw i8, ptr %0, i64 40
  %18 = load ptr, ptr %17, align 8
  tail call void @free(ptr noundef %18)
  %19 = icmp eq i32 %1, 0
  br i1 %19, label %21, label %20

20:                                               ; preds = %16
  tail call void @"??3@YAXPEAX_K@Z"(ptr noundef %0, i64 noundef 48) #28
  br label %21

21:                                               ; preds = %20, %16
  ret ptr %0
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @"?_Incref@facet@locale@std@@UEAAXXZ"(ptr noundef nonnull align 8 dereferenceable(16) %0) unnamed_addr #7 comdat align 2 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %3 = atomicrmw add ptr %2, i32 1 seq_cst, align 8
  ret void
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local noundef ptr @"?_Decref@facet@locale@std@@UEAAPEAV_Facet_base@3@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %0) unnamed_addr #7 comdat align 2 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %3 = atomicrmw sub ptr %2, i32 1 seq_cst, align 8
  %4 = icmp eq i32 %3, 1
  %5 = select i1 %4, ptr %0, ptr null
  ret ptr %5
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef ptr @"?do_tolower@?$ctype@D@std@@MEBAPEBDPEADPEBD@Z"(ptr noundef nonnull align 8 dereferenceable(48) %0, ptr noundef %1, ptr noundef %2) unnamed_addr #2 comdat align 2 {
  %4 = icmp eq ptr %1, %2
  br i1 %4, label %15, label %5

5:                                                ; preds = %3
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 16
  br label %7

7:                                                ; preds = %5, %7
  %8 = phi ptr [ %1, %5 ], [ %13, %7 ]
  %9 = load i8, ptr %8, align 1
  %10 = zext i8 %9 to i32
  %11 = tail call i32 @_Tolower(i32 noundef %10, ptr noundef nonnull %6) #10
  %12 = trunc i32 %11 to i8
  store i8 %12, ptr %8, align 1
  %13 = getelementptr inbounds nuw i8, ptr %8, i64 1
  %14 = icmp eq ptr %13, %2
  br i1 %14, label %15, label %7, !llvm.loop !40

15:                                               ; preds = %7, %3
  %16 = phi ptr [ %1, %3 ], [ %13, %7 ]
  ret ptr %16
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local noundef i8 @"?do_tolower@?$ctype@D@std@@MEBADD@Z"(ptr noundef nonnull align 8 dereferenceable(48) %0, i8 noundef %1) unnamed_addr #7 comdat align 2 {
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %4 = zext i8 %1 to i32
  %5 = tail call i32 @_Tolower(i32 noundef %4, ptr noundef nonnull %3) #10
  %6 = trunc i32 %5 to i8
  ret i8 %6
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local noundef ptr @"?do_toupper@?$ctype@D@std@@MEBAPEBDPEADPEBD@Z"(ptr noundef nonnull align 8 dereferenceable(48) %0, ptr noundef %1, ptr noundef %2) unnamed_addr #7 comdat align 2 {
  %4 = icmp eq ptr %1, %2
  br i1 %4, label %15, label %5

5:                                                ; preds = %3
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 16
  br label %7

7:                                                ; preds = %5, %7
  %8 = phi ptr [ %1, %5 ], [ %13, %7 ]
  %9 = load i8, ptr %8, align 1
  %10 = zext i8 %9 to i32
  %11 = tail call i32 @_Toupper(i32 noundef %10, ptr noundef nonnull %6) #10
  %12 = trunc i32 %11 to i8
  store i8 %12, ptr %8, align 1
  %13 = getelementptr inbounds nuw i8, ptr %8, i64 1
  %14 = icmp eq ptr %13, %2
  br i1 %14, label %15, label %7, !llvm.loop !41

15:                                               ; preds = %7, %3
  %16 = phi ptr [ %1, %3 ], [ %13, %7 ]
  ret ptr %16
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local noundef i8 @"?do_toupper@?$ctype@D@std@@MEBADD@Z"(ptr noundef nonnull align 8 dereferenceable(48) %0, i8 noundef %1) unnamed_addr #7 comdat align 2 {
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %4 = zext i8 %1 to i32
  %5 = tail call i32 @_Toupper(i32 noundef %4, ptr noundef nonnull %3) #10
  %6 = trunc i32 %5 to i8
  ret i8 %6
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef ptr @"?do_widen@?$ctype@D@std@@MEBAPEBDPEBD0PEAD@Z"(ptr noundef nonnull align 8 dereferenceable(48) %0, ptr noundef %1, ptr noundef %2, ptr noundef %3) unnamed_addr #2 comdat align 2 {
  %5 = ptrtoint ptr %2 to i64
  %6 = ptrtoint ptr %1 to i64
  %7 = sub i64 %5, %6
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 1 %3, ptr align 1 %1, i64 %7, i1 false)
  ret ptr %2
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local noundef i8 @"?do_widen@?$ctype@D@std@@MEBADD@Z"(ptr noundef nonnull align 8 dereferenceable(48) %0, i8 noundef %1) unnamed_addr #7 comdat align 2 {
  ret i8 %1
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local noundef ptr @"?do_narrow@?$ctype@D@std@@MEBAPEBDPEBD0DPEAD@Z"(ptr noundef nonnull align 8 dereferenceable(48) %0, ptr noundef %1, ptr noundef %2, i8 noundef %3, ptr noundef %4) unnamed_addr #7 comdat align 2 {
  %6 = ptrtoint ptr %2 to i64
  %7 = ptrtoint ptr %1 to i64
  %8 = sub i64 %6, %7
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 1 %4, ptr align 1 %1, i64 %8, i1 false)
  ret ptr %2
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local noundef i8 @"?do_narrow@?$ctype@D@std@@MEBADDD@Z"(ptr noundef nonnull align 8 dereferenceable(48) %0, i8 noundef %1, i8 noundef %2) unnamed_addr #7 comdat align 2 {
  ret i8 %1
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local noundef ptr @"??_Gctype_base@std@@UEAAPEAXI@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, i32 noundef %1) unnamed_addr #7 comdat align 2 {
  %3 = icmp eq i32 %1, 0
  br i1 %3, label %5, label %4

4:                                                ; preds = %2
  tail call void @"??3@YAXPEAX_K@Z"(ptr noundef %0, i64 noundef 16) #28
  br label %5

5:                                                ; preds = %4, %2
  ret ptr %0
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local noundef ptr @"??_Gfacet@locale@std@@MEAAPEAXI@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, i32 noundef %1) unnamed_addr #7 comdat align 2 {
  %3 = icmp eq i32 %1, 0
  br i1 %3, label %5, label %4

4:                                                ; preds = %2
  tail call void @"??3@YAXPEAX_K@Z"(ptr noundef %0, i64 noundef 16) #28
  br label %5

5:                                                ; preds = %4, %2
  ret ptr %0
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local noundef ptr @"??_G_Facet_base@std@@UEAAPEAXI@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0, i32 noundef %1) unnamed_addr #7 comdat align 2 {
  tail call void @llvm.trap() #30
  unreachable
}

declare dso_local void @_purecall() unnamed_addr

; Function Attrs: cold noreturn nounwind memory(inaccessiblemem: write)
declare void @llvm.trap() #18

; Function Attrs: nounwind
declare dso_local void @_Getctype(ptr dead_on_unwind writable sret(%struct._Ctypevec) align 8) local_unnamed_addr #16

; Function Attrs: nobuiltin nounwind
declare dso_local void @"??_V@YAXPEAX@Z"(ptr noundef) local_unnamed_addr #5

; Function Attrs: nounwind
declare dso_local i32 @_Tolower(i32 noundef, ptr noundef) local_unnamed_addr #16

; Function Attrs: nounwind
declare dso_local i32 @_Toupper(i32 noundef, ptr noundef) local_unnamed_addr #16

declare dso_local void @"?_Locinfo_dtor@_Locinfo@std@@SAXPEAV12@@Z"(ptr noundef) local_unnamed_addr #11

; Function Attrs: inlinehint mustprogress nounwind uwtable
define linkonce_odr dso_local noundef ptr @"??0bad_cast@std@@QEAA@AEBV01@@Z"(ptr noundef nonnull returned align 8 dereferenceable(24) %0, ptr noundef nonnull align 8 dereferenceable(24) %1) unnamed_addr #9 comdat align 2 personality ptr @__CxxFrameHandler3 {
  store ptr @"??_7exception@std@@6B@", ptr %0, align 8
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %3, i8 0, i64 16, i1 false)
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  invoke void @__std_exception_copy(ptr noundef nonnull %4, ptr noundef nonnull %3)
          to label %7 unwind label %5

5:                                                ; preds = %2
  %6 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %6) ]
  unreachable

7:                                                ; preds = %2
  store ptr @"??_7bad_cast@std@@6B@", ptr %0, align 8
  ret ptr %0
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @"??1exception@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #7 comdat align 2 personality ptr @__CxxFrameHandler3 {
  store ptr @"??_7exception@std@@6B@", ptr %0, align 8
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 8
  invoke void @__std_exception_destroy(ptr noundef nonnull %2)
          to label %3 unwind label %4

3:                                                ; preds = %1
  ret void

4:                                                ; preds = %1
  %5 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %5) ]
  unreachable
}

; Function Attrs: inlinehint mustprogress nounwind uwtable
define linkonce_odr dso_local noundef ptr @"??_Gbad_cast@std@@UEAAPEAXI@Z"(ptr noundef nonnull align 8 dereferenceable(24) %0, i32 noundef %1) unnamed_addr #9 comdat align 2 personality ptr @__CxxFrameHandler3 {
  store ptr @"??_7exception@std@@6B@", ptr %0, align 8
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  invoke void @__std_exception_destroy(ptr noundef nonnull %3)
          to label %6 unwind label %4

4:                                                ; preds = %2
  %5 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %5) ]
  unreachable

6:                                                ; preds = %2
  %7 = icmp eq i32 %1, 0
  br i1 %7, label %9, label %8

8:                                                ; preds = %6
  tail call void @"??3@YAXPEAX_K@Z"(ptr noundef %0, i64 noundef 24) #28
  br label %9

9:                                                ; preds = %8, %6
  ret ptr %0
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(16) ptr @"??$use_facet@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0) local_unnamed_addr #2 comdat personality ptr @__CxxFrameHandler3 {
  %2 = alloca %"class.std::_Locinfo", align 8
  %3 = alloca %"class.std::_Lockit", align 4
  %4 = alloca %"class.std::_Lockit", align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %4) #10
  %5 = call noundef ptr @"??0_Lockit@std@@QEAA@H@Z"(ptr noundef nonnull align 4 dereferenceable(4) %4, i32 noundef 0) #10
  %6 = load ptr, ptr @"?_Psave@?$_Facetptr@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@2PEBVfacet@locale@2@EB", align 8
  %7 = load i64, ptr @"?id@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A", align 8
  %8 = icmp eq i64 %7, 0
  br i1 %8, label %9, label %18

9:                                                ; preds = %1
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %3) #10
  %10 = call noundef ptr @"??0_Lockit@std@@QEAA@H@Z"(ptr noundef nonnull align 4 dereferenceable(4) %3, i32 noundef 0) #10
  %11 = load i64, ptr @"?id@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A", align 8
  %12 = icmp eq i64 %11, 0
  br i1 %12, label %13, label %17

13:                                               ; preds = %9
  %14 = load i32, ptr @"?_Id_cnt@id@locale@std@@0HA", align 4
  %15 = add nsw i32 %14, 1
  store i32 %15, ptr @"?_Id_cnt@id@locale@std@@0HA", align 4
  %16 = sext i32 %15 to i64
  store i64 %16, ptr @"?id@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A", align 8
  br label %17

17:                                               ; preds = %13, %9
  call void @"??1_Lockit@std@@QEAA@XZ"(ptr noundef nonnull align 4 dereferenceable(4) %3) #10
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %3) #10
  br label %18

18:                                               ; preds = %1, %17
  %19 = load i64, ptr @"?id@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A", align 8
  %20 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %21 = load ptr, ptr %20, align 8
  %22 = getelementptr inbounds nuw i8, ptr %21, i64 24
  %23 = load i64, ptr %22, align 8
  %24 = icmp ult i64 %19, %23
  br i1 %24, label %25, label %30

25:                                               ; preds = %18
  %26 = getelementptr inbounds nuw i8, ptr %21, i64 16
  %27 = load ptr, ptr %26, align 8
  %28 = getelementptr inbounds nuw ptr, ptr %27, i64 %19
  %29 = load ptr, ptr %28, align 8
  br label %30

30:                                               ; preds = %25, %18
  %31 = phi ptr [ %29, %25 ], [ null, %18 ]
  %32 = icmp eq ptr %31, null
  br i1 %32, label %33, label %48

33:                                               ; preds = %30
  %34 = getelementptr inbounds nuw i8, ptr %21, i64 36
  %35 = load i8, ptr %34, align 4, !range !18, !noundef !19
  %36 = trunc nuw i8 %35 to i1
  br i1 %36, label %37, label %48

37:                                               ; preds = %33
  %38 = invoke noundef ptr @"?_Getgloballocale@locale@std@@CAPEAV_Locimp@12@XZ"()
          to label %39 unwind label %82

39:                                               ; preds = %37
  %40 = getelementptr inbounds nuw i8, ptr %38, i64 24
  %41 = load i64, ptr %40, align 8
  %42 = icmp ult i64 %19, %41
  br i1 %42, label %43, label %48

43:                                               ; preds = %39
  %44 = getelementptr inbounds nuw i8, ptr %38, i64 16
  %45 = load ptr, ptr %44, align 8
  %46 = getelementptr inbounds nuw ptr, ptr %45, i64 %19
  %47 = load ptr, ptr %46, align 8
  br label %48

48:                                               ; preds = %43, %39, %33, %30
  %49 = phi ptr [ %31, %33 ], [ %31, %30 ], [ %47, %43 ], [ null, %39 ]
  %50 = icmp eq ptr %49, null
  br i1 %50, label %51, label %80

51:                                               ; preds = %48
  %52 = icmp eq ptr %6, null
  br i1 %52, label %53, label %80

53:                                               ; preds = %51
  %54 = invoke noalias noundef nonnull dereferenceable(16) ptr @"??2@YAPEAX_K@Z"(i64 noundef 16) #27
          to label %55 unwind label %82

55:                                               ; preds = %53
  call void @llvm.lifetime.start.p0(i64 104, ptr nonnull %2) #10
  %56 = load ptr, ptr %20, align 8
  %57 = icmp eq ptr %56, null
  br i1 %57, label %64, label %58

58:                                               ; preds = %55
  %59 = getelementptr inbounds nuw i8, ptr %56, i64 40
  %60 = load ptr, ptr %59, align 8
  %61 = icmp eq ptr %60, null
  %62 = getelementptr inbounds nuw i8, ptr %56, i64 48
  %63 = select i1 %61, ptr %62, ptr %60
  br label %64

64:                                               ; preds = %58, %55
  %65 = phi ptr [ %63, %58 ], [ @"??_C@_00CNPNBAHC@?$AA@", %55 ]
  %66 = invoke noundef ptr @"??0_Locinfo@std@@QEAA@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(104) %2, ptr noundef %65)
          to label %67 unwind label %69

67:                                               ; preds = %64
  store ptr @"??_7facet@locale@std@@6B@", ptr %54, align 8
  %68 = getelementptr inbounds nuw i8, ptr %54, i64 8
  store i32 0, ptr %68, align 8
  store ptr @"??_7?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@6B@", ptr %54, align 8
  call void @"??1_Locinfo@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(104) %2) #10
  call void @llvm.lifetime.end.p0(i64 104, ptr nonnull %2) #10
  invoke void @"?_Facet_Register@std@@YAXPEAV_Facet_base@1@@Z"(ptr noundef nonnull %54)
          to label %71 unwind label %75

69:                                               ; preds = %64
  %70 = cleanuppad within none []
  call void @llvm.lifetime.end.p0(i64 104, ptr nonnull %2) #10
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %54, i64 noundef 16) #28 [ "funclet"(token %70) ]
  cleanupret from %70 unwind label %82

71:                                               ; preds = %67
  %72 = load ptr, ptr %54, align 8
  %73 = getelementptr inbounds nuw i8, ptr %72, i64 8
  %74 = load ptr, ptr %73, align 8
  call void %74(ptr noundef nonnull align 8 dereferenceable(16) %54) #10
  store ptr %54, ptr @"?_Psave@?$_Facetptr@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@2PEBVfacet@locale@2@EB", align 8
  br label %80

75:                                               ; preds = %67
  %76 = cleanuppad within none []
  %77 = load ptr, ptr %54, align 8
  %78 = load ptr, ptr %77, align 8
  %79 = call noundef ptr %78(ptr noundef nonnull align 8 dereferenceable(8) %54, i32 noundef 1) #10 [ "funclet"(token %76) ]
  cleanupret from %76 unwind label %82

80:                                               ; preds = %51, %71, %48
  %81 = phi ptr [ %49, %48 ], [ %54, %71 ], [ %6, %51 ]
  call void @"??1_Lockit@std@@QEAA@XZ"(ptr noundef nonnull align 4 dereferenceable(4) %4) #10
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %4) #10
  ret ptr %81

82:                                               ; preds = %53, %69, %37, %75
  %83 = cleanuppad within none []
  call void @"??1_Lockit@std@@QEAA@XZ"(ptr noundef nonnull align 4 dereferenceable(4) %4) #10 [ "funclet"(token %83) ]
  cleanupret from %83 unwind to caller
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local noundef ptr @"??_G?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAAPEAXI@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, i32 noundef %1) unnamed_addr #7 comdat align 2 {
  %3 = icmp eq i32 %1, 0
  br i1 %3, label %5, label %4

4:                                                ; preds = %2
  tail call void @"??3@YAXPEAX_K@Z"(ptr noundef %0, i64 noundef 16) #28
  br label %5

5:                                                ; preds = %4, %2
  ret ptr %0
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBX@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr dead_on_unwind noalias writable sret(%"class.std::ostreambuf_iterator") align 8 %1, ptr dead_on_return noundef %2, ptr noundef nonnull align 8 dereferenceable(72) %3, i8 noundef %4, ptr noundef %5) unnamed_addr #2 comdat align 2 {
  %7 = alloca [64 x i8], align 16
  %8 = alloca %"class.std::ostreambuf_iterator", align 8
  call void @llvm.lifetime.start.p0(i64 64, ptr nonnull %7) #10
  %9 = call i32 (ptr, i64, ptr, ...) @sprintf_s(ptr noundef nonnull %7, i64 noundef 64, ptr noundef nonnull @"??_C@_02BBAHNLBA@?$CFp?$AA@", ptr noundef %5)
  %10 = sext i32 %9 to i64
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %8, ptr noundef nonnull align 8 dereferenceable(16) %2, i64 16, i1 false)
  call void @"?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr dead_on_unwind writable sret(%"class.std::ostreambuf_iterator") align 8 %1, ptr dead_on_return noundef nonnull %8, ptr noundef nonnull align 8 dereferenceable(72) %3, i8 noundef %4, ptr noundef nonnull %7, i64 noundef %10)
  call void @llvm.lifetime.end.p0(i64 64, ptr nonnull %7) #10
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DO@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr dead_on_unwind noalias writable sret(%"class.std::ostreambuf_iterator") align 8 %1, ptr dead_on_return noundef %2, ptr noundef nonnull align 8 dereferenceable(72) %3, i8 noundef %4, double noundef %5) unnamed_addr #2 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %7 = alloca %"class.std::basic_string", align 8
  %8 = alloca [8 x i8], align 1
  %9 = alloca i32, align 4
  %10 = alloca %"class.std::ostreambuf_iterator", align 8
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %7) #10
  %11 = getelementptr inbounds nuw i8, ptr %7, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %7, i8 0, i64 32, i1 false)
  store i64 15, ptr %11, align 8
  store i8 0, ptr %7, align 8
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %8) #10
  %12 = getelementptr inbounds nuw i8, ptr %3, i64 24
  %13 = load i32, ptr %12, align 8
  %14 = and i32 %13, 12288
  %15 = icmp eq i32 %14, 8192
  %16 = icmp eq i32 %14, 12288
  %17 = getelementptr inbounds nuw i8, ptr %3, i64 32
  %18 = load i64, ptr %17, align 8
  %19 = select i1 %16, i64 -1, i64 %18
  br i1 %16, label %27, label %20

20:                                               ; preds = %6
  %21 = icmp sgt i64 %19, 0
  br i1 %21, label %27, label %22

22:                                               ; preds = %20
  %23 = icmp eq i64 %19, 0
  br i1 %23, label %24, label %27

24:                                               ; preds = %22
  %25 = icmp eq i32 %14, 0
  %26 = zext i1 %25 to i64
  br label %27

27:                                               ; preds = %20, %6, %22, %24
  %28 = phi i64 [ %26, %24 ], [ 13, %6 ], [ 6, %22 ], [ %19, %20 ]
  %29 = shl i64 %28, 32
  %30 = ashr exact i64 %29, 32
  %31 = tail call double @llvm.fabs.f64(double %5)
  %32 = fcmp ogt double %31, 1.000000e+10
  %33 = and i1 %32, %15
  br i1 %33, label %34, label %42

34:                                               ; preds = %27
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %9) #10
  %35 = call double @frexp(double noundef %5, ptr noundef nonnull %9) #10
  %36 = load i32, ptr %9, align 4
  %37 = tail call i32 @llvm.abs.i32(i32 %36, i1 true)
  %38 = mul nuw nsw i32 %37, 30103
  %39 = udiv i32 %38, 100000
  %40 = zext nneg i32 %39 to i64
  %41 = add nsw i64 %30, %40
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %9) #10
  br label %42

42:                                               ; preds = %34, %27
  %43 = phi i64 [ %41, %34 ], [ %30, %27 ]
  %44 = add nsw i64 %43, 50
  %45 = getelementptr inbounds nuw i8, ptr %7, i64 16
  %46 = load i64, ptr %45, align 8
  %47 = icmp ugt i64 %44, %46
  br i1 %47, label %49, label %48

48:                                               ; preds = %42
  store i64 %44, ptr %45, align 8
  br label %57

49:                                               ; preds = %42
  %50 = sub nuw i64 %44, %46
  %51 = sub i64 15, %46
  %52 = icmp ugt i64 %50, %51
  br i1 %52, label %55, label %53

53:                                               ; preds = %49
  store i64 %44, ptr %45, align 8
  %54 = getelementptr inbounds nuw i8, ptr %7, i64 %46
  call void @llvm.memset.p0.i64(ptr nonnull align 1 %54, i8 0, i64 %50, i1 false)
  br label %57

55:                                               ; preds = %49
  %56 = invoke noundef nonnull align 8 dereferenceable(32) ptr @"??$_Reallocate_grow_by@V<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@_KD@Z@_KD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??append@01@QEAAAEAV01@0D@Z@_KD@Z"(ptr noundef nonnull align 8 dereferenceable(32) %7, i64 noundef %50, i8 undef, i64 noundef %50, i8 noundef 0)
          to label %59 unwind label %126

57:                                               ; preds = %48, %53
  %58 = getelementptr inbounds nuw i8, ptr %7, i64 %44
  store i8 0, ptr %58, align 1
  br label %59

59:                                               ; preds = %57, %55
  %60 = fcmp one double %31, 0x7FF0000000000000
  %61 = load i32, ptr %12, align 8
  %62 = and i32 %61, -17
  %63 = select i1 %60, i32 %61, i32 %62
  %64 = trunc i64 %19 to i32
  %65 = getelementptr inbounds nuw i8, ptr %8, i64 1
  store i8 37, ptr %8, align 1
  %66 = and i32 %63, 32
  %67 = icmp eq i32 %66, 0
  br i1 %67, label %70, label %68

68:                                               ; preds = %59
  %69 = getelementptr inbounds nuw i8, ptr %8, i64 2
  store i8 43, ptr %65, align 1
  br label %70

70:                                               ; preds = %68, %59
  %71 = phi ptr [ %69, %68 ], [ %65, %59 ]
  %72 = and i32 %63, 16
  %73 = icmp eq i32 %72, 0
  br i1 %73, label %76, label %74

74:                                               ; preds = %70
  %75 = getelementptr inbounds nuw i8, ptr %71, i64 1
  store i8 35, ptr %71, align 1
  br label %76

76:                                               ; preds = %74, %70
  %77 = phi ptr [ %75, %74 ], [ %71, %70 ]
  %78 = getelementptr inbounds nuw i8, ptr %77, i64 1
  store i8 46, ptr %77, align 1
  %79 = getelementptr inbounds nuw i8, ptr %77, i64 2
  store i8 42, ptr %78, align 1
  %80 = getelementptr inbounds nuw i8, ptr %77, i64 3
  store i8 76, ptr %79, align 1
  %81 = and i32 %63, 4
  %82 = icmp eq i32 %81, 0
  %83 = select i1 %82, i32 1634100583, i32 1097221447
  %84 = lshr i32 %63, 9
  %85 = and i32 %84, 24
  %86 = lshr i32 %83, %85
  %87 = trunc i32 %86 to i8
  %88 = getelementptr inbounds nuw i8, ptr %77, i64 4
  store i8 %87, ptr %80, align 1
  store i8 0, ptr %88, align 1
  %89 = load i64, ptr %45, align 8
  %90 = load i64, ptr %11, align 8
  %91 = icmp ugt i64 %90, 15
  %92 = load ptr, ptr %7, align 8
  %93 = select i1 %91, ptr %92, ptr %7
  %94 = invoke i32 (ptr, i64, ptr, ...) @sprintf_s(ptr noundef nonnull %93, i64 noundef %89, ptr noundef nonnull %8, i32 noundef %64, double noundef %5)
          to label %95 unwind label %126

95:                                               ; preds = %76
  %96 = sext i32 %94 to i64
  %97 = load i64, ptr %11, align 8
  %98 = icmp ugt i64 %97, 15
  %99 = load ptr, ptr %7, align 8
  %100 = select i1 %98, ptr %99, ptr %7
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %10, ptr noundef nonnull align 8 dereferenceable(16) %2, i64 16, i1 false)
  invoke void @"??$_Fput_v3@$0A@@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@1@V21@AEAVios_base@1@DPEBD_K_N@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr dead_on_unwind writable sret(%"class.std::ostreambuf_iterator") align 8 %1, ptr dead_on_return noundef nonnull %10, ptr noundef nonnull align 8 dereferenceable(72) %3, i8 noundef %4, ptr noundef %100, i64 noundef %96, i1 noundef zeroext %60)
          to label %101 unwind label %126

101:                                              ; preds = %95
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %8) #10
  %102 = load i64, ptr %11, align 8
  %103 = icmp ugt i64 %102, 15
  br i1 %103, label %104, label %125

104:                                              ; preds = %101
  %105 = load ptr, ptr %7, align 8
  %106 = add i64 %102, 1
  %107 = icmp ugt i64 %106, 4095
  br i1 %107, label %108, label %122

108:                                              ; preds = %104
  %109 = getelementptr inbounds i8, ptr %105, i64 -8
  %110 = load i64, ptr %109, align 8
  %111 = ptrtoint ptr %105 to i64
  %112 = add i64 %111, -8
  %113 = sub i64 %112, %110
  %114 = icmp ult i64 %113, 32
  br i1 %114, label %117, label %115

115:                                              ; preds = %108
  invoke void @_invalid_parameter_noinfo_noreturn() #29
          to label %116 unwind label %120

116:                                              ; preds = %115
  unreachable

117:                                              ; preds = %108
  %118 = add i64 %102, 40
  %119 = inttoptr i64 %110 to ptr
  br label %122

120:                                              ; preds = %115
  %121 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %121) ]
  unreachable

122:                                              ; preds = %117, %104
  %123 = phi i64 [ %118, %117 ], [ %106, %104 ]
  %124 = phi ptr [ %119, %117 ], [ %105, %104 ]
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %124, i64 noundef %123) #10
  br label %125

125:                                              ; preds = %101, %122
  store i64 0, ptr %45, align 8
  store i64 15, ptr %11, align 8
  store i8 0, ptr %7, align 8
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %7) #10
  ret void

126:                                              ; preds = %55, %76, %95
  %127 = cleanuppad within none []
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %8) #10
  call void @"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %7) #10 [ "funclet"(token %127) ]
  cleanupret from %127 unwind to caller
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DN@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr dead_on_unwind noalias writable sret(%"class.std::ostreambuf_iterator") align 8 %1, ptr dead_on_return noundef %2, ptr noundef nonnull align 8 dereferenceable(72) %3, i8 noundef %4, double noundef %5) unnamed_addr #2 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %7 = alloca %"class.std::basic_string", align 8
  %8 = alloca [8 x i8], align 1
  %9 = alloca i32, align 4
  %10 = alloca %"class.std::ostreambuf_iterator", align 8
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %7) #10
  %11 = getelementptr inbounds nuw i8, ptr %7, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %7, i8 0, i64 32, i1 false)
  store i64 15, ptr %11, align 8
  store i8 0, ptr %7, align 8
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %8) #10
  %12 = getelementptr inbounds nuw i8, ptr %3, i64 24
  %13 = load i32, ptr %12, align 8
  %14 = and i32 %13, 12288
  %15 = icmp eq i32 %14, 8192
  %16 = icmp eq i32 %14, 12288
  %17 = getelementptr inbounds nuw i8, ptr %3, i64 32
  %18 = load i64, ptr %17, align 8
  %19 = select i1 %16, i64 -1, i64 %18
  br i1 %16, label %27, label %20

20:                                               ; preds = %6
  %21 = icmp sgt i64 %19, 0
  br i1 %21, label %27, label %22

22:                                               ; preds = %20
  %23 = icmp eq i64 %19, 0
  br i1 %23, label %24, label %27

24:                                               ; preds = %22
  %25 = icmp eq i32 %14, 0
  %26 = zext i1 %25 to i64
  br label %27

27:                                               ; preds = %20, %6, %22, %24
  %28 = phi i64 [ %26, %24 ], [ 13, %6 ], [ 6, %22 ], [ %19, %20 ]
  %29 = shl i64 %28, 32
  %30 = ashr exact i64 %29, 32
  %31 = tail call double @llvm.fabs.f64(double %5)
  %32 = fcmp ogt double %31, 1.000000e+10
  %33 = and i1 %32, %15
  br i1 %33, label %34, label %42

34:                                               ; preds = %27
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %9) #10
  %35 = call double @frexp(double noundef %5, ptr noundef nonnull %9) #10
  %36 = load i32, ptr %9, align 4
  %37 = tail call i32 @llvm.abs.i32(i32 %36, i1 true)
  %38 = mul nuw nsw i32 %37, 30103
  %39 = udiv i32 %38, 100000
  %40 = zext nneg i32 %39 to i64
  %41 = add nsw i64 %30, %40
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %9) #10
  br label %42

42:                                               ; preds = %34, %27
  %43 = phi i64 [ %41, %34 ], [ %30, %27 ]
  %44 = add nsw i64 %43, 50
  %45 = getelementptr inbounds nuw i8, ptr %7, i64 16
  %46 = load i64, ptr %45, align 8
  %47 = icmp ugt i64 %44, %46
  br i1 %47, label %49, label %48

48:                                               ; preds = %42
  store i64 %44, ptr %45, align 8
  br label %57

49:                                               ; preds = %42
  %50 = sub nuw i64 %44, %46
  %51 = sub i64 15, %46
  %52 = icmp ugt i64 %50, %51
  br i1 %52, label %55, label %53

53:                                               ; preds = %49
  store i64 %44, ptr %45, align 8
  %54 = getelementptr inbounds nuw i8, ptr %7, i64 %46
  call void @llvm.memset.p0.i64(ptr nonnull align 1 %54, i8 0, i64 %50, i1 false)
  br label %57

55:                                               ; preds = %49
  %56 = invoke noundef nonnull align 8 dereferenceable(32) ptr @"??$_Reallocate_grow_by@V<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@_KD@Z@_KD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??append@01@QEAAAEAV01@0D@Z@_KD@Z"(ptr noundef nonnull align 8 dereferenceable(32) %7, i64 noundef %50, i8 undef, i64 noundef %50, i8 noundef 0)
          to label %59 unwind label %125

57:                                               ; preds = %48, %53
  %58 = getelementptr inbounds nuw i8, ptr %7, i64 %44
  store i8 0, ptr %58, align 1
  br label %59

59:                                               ; preds = %57, %55
  %60 = fcmp one double %31, 0x7FF0000000000000
  %61 = load i32, ptr %12, align 8
  %62 = and i32 %61, -17
  %63 = select i1 %60, i32 %61, i32 %62
  %64 = trunc i64 %19 to i32
  %65 = getelementptr inbounds nuw i8, ptr %8, i64 1
  store i8 37, ptr %8, align 1
  %66 = and i32 %63, 32
  %67 = icmp eq i32 %66, 0
  br i1 %67, label %70, label %68

68:                                               ; preds = %59
  %69 = getelementptr inbounds nuw i8, ptr %8, i64 2
  store i8 43, ptr %65, align 1
  br label %70

70:                                               ; preds = %68, %59
  %71 = phi ptr [ %69, %68 ], [ %65, %59 ]
  %72 = and i32 %63, 16
  %73 = icmp eq i32 %72, 0
  br i1 %73, label %76, label %74

74:                                               ; preds = %70
  %75 = getelementptr inbounds nuw i8, ptr %71, i64 1
  store i8 35, ptr %71, align 1
  br label %76

76:                                               ; preds = %74, %70
  %77 = phi ptr [ %75, %74 ], [ %71, %70 ]
  %78 = getelementptr inbounds nuw i8, ptr %77, i64 1
  store i8 46, ptr %77, align 1
  %79 = getelementptr inbounds nuw i8, ptr %77, i64 2
  store i8 42, ptr %78, align 1
  %80 = and i32 %63, 4
  %81 = icmp eq i32 %80, 0
  %82 = select i1 %81, i32 1634100583, i32 1097221447
  %83 = lshr i32 %63, 9
  %84 = and i32 %83, 24
  %85 = lshr i32 %82, %84
  %86 = trunc i32 %85 to i8
  %87 = getelementptr inbounds nuw i8, ptr %77, i64 3
  store i8 %86, ptr %79, align 1
  store i8 0, ptr %87, align 1
  %88 = load i64, ptr %45, align 8
  %89 = load i64, ptr %11, align 8
  %90 = icmp ugt i64 %89, 15
  %91 = load ptr, ptr %7, align 8
  %92 = select i1 %90, ptr %91, ptr %7
  %93 = invoke i32 (ptr, i64, ptr, ...) @sprintf_s(ptr noundef nonnull %92, i64 noundef %88, ptr noundef nonnull %8, i32 noundef %64, double noundef %5)
          to label %94 unwind label %125

94:                                               ; preds = %76
  %95 = sext i32 %93 to i64
  %96 = load i64, ptr %11, align 8
  %97 = icmp ugt i64 %96, 15
  %98 = load ptr, ptr %7, align 8
  %99 = select i1 %97, ptr %98, ptr %7
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %10, ptr noundef nonnull align 8 dereferenceable(16) %2, i64 16, i1 false)
  invoke void @"??$_Fput_v3@$0A@@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@1@V21@AEAVios_base@1@DPEBD_K_N@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr dead_on_unwind writable sret(%"class.std::ostreambuf_iterator") align 8 %1, ptr dead_on_return noundef nonnull %10, ptr noundef nonnull align 8 dereferenceable(72) %3, i8 noundef %4, ptr noundef %99, i64 noundef %95, i1 noundef zeroext %60)
          to label %100 unwind label %125

100:                                              ; preds = %94
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %8) #10
  %101 = load i64, ptr %11, align 8
  %102 = icmp ugt i64 %101, 15
  br i1 %102, label %103, label %124

103:                                              ; preds = %100
  %104 = load ptr, ptr %7, align 8
  %105 = add i64 %101, 1
  %106 = icmp ugt i64 %105, 4095
  br i1 %106, label %107, label %121

107:                                              ; preds = %103
  %108 = getelementptr inbounds i8, ptr %104, i64 -8
  %109 = load i64, ptr %108, align 8
  %110 = ptrtoint ptr %104 to i64
  %111 = add i64 %110, -8
  %112 = sub i64 %111, %109
  %113 = icmp ult i64 %112, 32
  br i1 %113, label %116, label %114

114:                                              ; preds = %107
  invoke void @_invalid_parameter_noinfo_noreturn() #29
          to label %115 unwind label %119

115:                                              ; preds = %114
  unreachable

116:                                              ; preds = %107
  %117 = add i64 %101, 40
  %118 = inttoptr i64 %109 to ptr
  br label %121

119:                                              ; preds = %114
  %120 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %120) ]
  unreachable

121:                                              ; preds = %116, %103
  %122 = phi i64 [ %117, %116 ], [ %105, %103 ]
  %123 = phi ptr [ %118, %116 ], [ %104, %103 ]
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %123, i64 noundef %122) #10
  br label %124

124:                                              ; preds = %100, %121
  store i64 0, ptr %45, align 8
  store i64 15, ptr %11, align 8
  store i8 0, ptr %7, align 8
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %7) #10
  ret void

125:                                              ; preds = %55, %76, %94
  %126 = cleanuppad within none []
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %8) #10
  call void @"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %7) #10 [ "funclet"(token %126) ]
  cleanupret from %126 unwind to caller
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_K@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr dead_on_unwind noalias writable sret(%"class.std::ostreambuf_iterator") align 8 %1, ptr dead_on_return noundef %2, ptr noundef nonnull align 8 dereferenceable(72) %3, i8 noundef %4, i64 noundef %5) unnamed_addr #2 comdat align 2 {
  %7 = alloca [64 x i8], align 16
  %8 = alloca [8 x i8], align 1
  %9 = alloca %"class.std::ostreambuf_iterator", align 8
  call void @llvm.lifetime.start.p0(i64 64, ptr nonnull %7) #10
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %8) #10
  %10 = getelementptr inbounds nuw i8, ptr %3, i64 24
  %11 = load i32, ptr %10, align 8
  %12 = getelementptr inbounds nuw i8, ptr %8, i64 1
  store i8 37, ptr %8, align 1
  %13 = and i32 %11, 32
  %14 = icmp eq i32 %13, 0
  br i1 %14, label %17, label %15

15:                                               ; preds = %6
  %16 = getelementptr inbounds nuw i8, ptr %8, i64 2
  store i8 43, ptr %12, align 1
  br label %17

17:                                               ; preds = %15, %6
  %18 = phi ptr [ %16, %15 ], [ %12, %6 ]
  %19 = and i32 %11, 8
  %20 = icmp eq i32 %19, 0
  br i1 %20, label %23, label %21

21:                                               ; preds = %17
  %22 = getelementptr inbounds nuw i8, ptr %18, i64 1
  store i8 35, ptr %18, align 1
  br label %23

23:                                               ; preds = %21, %17
  %24 = phi ptr [ %22, %21 ], [ %18, %17 ]
  %25 = getelementptr inbounds nuw i8, ptr %24, i64 1
  store i8 73, ptr %24, align 1
  %26 = getelementptr inbounds nuw i8, ptr %24, i64 2
  store i8 54, ptr %25, align 1
  store i8 52, ptr %26, align 1
  %27 = and i32 %11, 3584
  switch i32 %27, label %28 [
    i32 1024, label %33
    i32 2048, label %29
  ]

28:                                               ; preds = %23
  br label %33

29:                                               ; preds = %23
  %30 = and i32 %11, 4
  %31 = icmp eq i32 %30, 0
  %32 = select i1 %31, i8 120, i8 88
  br label %33

33:                                               ; preds = %23, %28, %29
  %34 = phi i8 [ 111, %23 ], [ 117, %28 ], [ %32, %29 ]
  %35 = getelementptr inbounds nuw i8, ptr %24, i64 3
  %36 = getelementptr inbounds nuw i8, ptr %24, i64 4
  store i8 %34, ptr %35, align 1
  store i8 0, ptr %36, align 1
  %37 = call i32 (ptr, i64, ptr, ...) @sprintf_s(ptr noundef nonnull %7, i64 noundef 64, ptr noundef nonnull %8, i64 noundef %5)
  %38 = sext i32 %37 to i64
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %9, ptr noundef nonnull align 8 dereferenceable(16) %2, i64 16, i1 false)
  call void @"?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr dead_on_unwind writable sret(%"class.std::ostreambuf_iterator") align 8 %1, ptr dead_on_return noundef nonnull %9, ptr noundef nonnull align 8 dereferenceable(72) %3, i8 noundef %4, ptr noundef nonnull %7, i64 noundef %38)
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %8) #10
  call void @llvm.lifetime.end.p0(i64 64, ptr nonnull %7) #10
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_J@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr dead_on_unwind noalias writable sret(%"class.std::ostreambuf_iterator") align 8 %1, ptr dead_on_return noundef %2, ptr noundef nonnull align 8 dereferenceable(72) %3, i8 noundef %4, i64 noundef %5) unnamed_addr #2 comdat align 2 {
  %7 = alloca [64 x i8], align 16
  %8 = alloca [8 x i8], align 1
  %9 = alloca %"class.std::ostreambuf_iterator", align 8
  call void @llvm.lifetime.start.p0(i64 64, ptr nonnull %7) #10
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %8) #10
  %10 = getelementptr inbounds nuw i8, ptr %3, i64 24
  %11 = load i32, ptr %10, align 8
  %12 = getelementptr inbounds nuw i8, ptr %8, i64 1
  store i8 37, ptr %8, align 1
  %13 = and i32 %11, 32
  %14 = icmp eq i32 %13, 0
  br i1 %14, label %17, label %15

15:                                               ; preds = %6
  %16 = getelementptr inbounds nuw i8, ptr %8, i64 2
  store i8 43, ptr %12, align 1
  br label %17

17:                                               ; preds = %15, %6
  %18 = phi ptr [ %16, %15 ], [ %12, %6 ]
  %19 = and i32 %11, 8
  %20 = icmp eq i32 %19, 0
  br i1 %20, label %23, label %21

21:                                               ; preds = %17
  %22 = getelementptr inbounds nuw i8, ptr %18, i64 1
  store i8 35, ptr %18, align 1
  br label %23

23:                                               ; preds = %21, %17
  %24 = phi ptr [ %22, %21 ], [ %18, %17 ]
  %25 = getelementptr inbounds nuw i8, ptr %24, i64 1
  store i8 73, ptr %24, align 1
  %26 = getelementptr inbounds nuw i8, ptr %24, i64 2
  store i8 54, ptr %25, align 1
  store i8 52, ptr %26, align 1
  %27 = and i32 %11, 3584
  switch i32 %27, label %28 [
    i32 1024, label %33
    i32 2048, label %29
  ]

28:                                               ; preds = %23
  br label %33

29:                                               ; preds = %23
  %30 = and i32 %11, 4
  %31 = icmp eq i32 %30, 0
  %32 = select i1 %31, i8 120, i8 88
  br label %33

33:                                               ; preds = %23, %28, %29
  %34 = phi i8 [ 111, %23 ], [ 100, %28 ], [ %32, %29 ]
  %35 = getelementptr inbounds nuw i8, ptr %24, i64 3
  %36 = getelementptr inbounds nuw i8, ptr %24, i64 4
  store i8 %34, ptr %35, align 1
  store i8 0, ptr %36, align 1
  %37 = call i32 (ptr, i64, ptr, ...) @sprintf_s(ptr noundef nonnull %7, i64 noundef 64, ptr noundef nonnull %8, i64 noundef %5)
  %38 = sext i32 %37 to i64
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %9, ptr noundef nonnull align 8 dereferenceable(16) %2, i64 16, i1 false)
  call void @"?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr dead_on_unwind writable sret(%"class.std::ostreambuf_iterator") align 8 %1, ptr dead_on_return noundef nonnull %9, ptr noundef nonnull align 8 dereferenceable(72) %3, i8 noundef %4, ptr noundef nonnull %7, i64 noundef %38)
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %8) #10
  call void @llvm.lifetime.end.p0(i64 64, ptr nonnull %7) #10
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DK@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr dead_on_unwind noalias writable sret(%"class.std::ostreambuf_iterator") align 8 %1, ptr dead_on_return noundef %2, ptr noundef nonnull align 8 dereferenceable(72) %3, i8 noundef %4, i32 noundef %5) unnamed_addr #2 comdat align 2 {
  %7 = alloca [64 x i8], align 16
  %8 = alloca [6 x i8], align 1
  %9 = alloca %"class.std::ostreambuf_iterator", align 8
  call void @llvm.lifetime.start.p0(i64 64, ptr nonnull %7) #10
  call void @llvm.lifetime.start.p0(i64 6, ptr nonnull %8) #10
  %10 = getelementptr inbounds nuw i8, ptr %3, i64 24
  %11 = load i32, ptr %10, align 8
  %12 = getelementptr inbounds nuw i8, ptr %8, i64 1
  store i8 37, ptr %8, align 1
  %13 = and i32 %11, 32
  %14 = icmp eq i32 %13, 0
  br i1 %14, label %17, label %15

15:                                               ; preds = %6
  %16 = getelementptr inbounds nuw i8, ptr %8, i64 2
  store i8 43, ptr %12, align 1
  br label %17

17:                                               ; preds = %15, %6
  %18 = phi ptr [ %16, %15 ], [ %12, %6 ]
  %19 = and i32 %11, 8
  %20 = icmp eq i32 %19, 0
  br i1 %20, label %23, label %21

21:                                               ; preds = %17
  %22 = getelementptr inbounds nuw i8, ptr %18, i64 1
  store i8 35, ptr %18, align 1
  br label %23

23:                                               ; preds = %21, %17
  %24 = phi ptr [ %22, %21 ], [ %18, %17 ]
  store i8 108, ptr %24, align 1
  %25 = and i32 %11, 3584
  switch i32 %25, label %26 [
    i32 1024, label %31
    i32 2048, label %27
  ]

26:                                               ; preds = %23
  br label %31

27:                                               ; preds = %23
  %28 = and i32 %11, 4
  %29 = icmp eq i32 %28, 0
  %30 = select i1 %29, i8 120, i8 88
  br label %31

31:                                               ; preds = %23, %26, %27
  %32 = phi i8 [ 111, %23 ], [ 117, %26 ], [ %30, %27 ]
  %33 = getelementptr inbounds nuw i8, ptr %24, i64 1
  %34 = getelementptr inbounds nuw i8, ptr %24, i64 2
  store i8 %32, ptr %33, align 1
  store i8 0, ptr %34, align 1
  %35 = call i32 (ptr, i64, ptr, ...) @sprintf_s(ptr noundef nonnull %7, i64 noundef 64, ptr noundef nonnull %8, i32 noundef %5)
  %36 = sext i32 %35 to i64
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %9, ptr noundef nonnull align 8 dereferenceable(16) %2, i64 16, i1 false)
  call void @"?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr dead_on_unwind writable sret(%"class.std::ostreambuf_iterator") align 8 %1, ptr dead_on_return noundef nonnull %9, ptr noundef nonnull align 8 dereferenceable(72) %3, i8 noundef %4, ptr noundef nonnull %7, i64 noundef %36)
  call void @llvm.lifetime.end.p0(i64 6, ptr nonnull %8) #10
  call void @llvm.lifetime.end.p0(i64 64, ptr nonnull %7) #10
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DJ@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr dead_on_unwind noalias writable sret(%"class.std::ostreambuf_iterator") align 8 %1, ptr dead_on_return noundef %2, ptr noundef nonnull align 8 dereferenceable(72) %3, i8 noundef %4, i32 noundef %5) unnamed_addr #2 comdat align 2 {
  %7 = alloca [64 x i8], align 16
  %8 = alloca [6 x i8], align 1
  %9 = alloca %"class.std::ostreambuf_iterator", align 8
  call void @llvm.lifetime.start.p0(i64 64, ptr nonnull %7) #10
  call void @llvm.lifetime.start.p0(i64 6, ptr nonnull %8) #10
  %10 = getelementptr inbounds nuw i8, ptr %3, i64 24
  %11 = load i32, ptr %10, align 8
  %12 = getelementptr inbounds nuw i8, ptr %8, i64 1
  store i8 37, ptr %8, align 1
  %13 = and i32 %11, 32
  %14 = icmp eq i32 %13, 0
  br i1 %14, label %17, label %15

15:                                               ; preds = %6
  %16 = getelementptr inbounds nuw i8, ptr %8, i64 2
  store i8 43, ptr %12, align 1
  br label %17

17:                                               ; preds = %15, %6
  %18 = phi ptr [ %16, %15 ], [ %12, %6 ]
  %19 = and i32 %11, 8
  %20 = icmp eq i32 %19, 0
  br i1 %20, label %23, label %21

21:                                               ; preds = %17
  %22 = getelementptr inbounds nuw i8, ptr %18, i64 1
  store i8 35, ptr %18, align 1
  br label %23

23:                                               ; preds = %21, %17
  %24 = phi ptr [ %22, %21 ], [ %18, %17 ]
  store i8 108, ptr %24, align 1
  %25 = and i32 %11, 3584
  switch i32 %25, label %26 [
    i32 1024, label %31
    i32 2048, label %27
  ]

26:                                               ; preds = %23
  br label %31

27:                                               ; preds = %23
  %28 = and i32 %11, 4
  %29 = icmp eq i32 %28, 0
  %30 = select i1 %29, i8 120, i8 88
  br label %31

31:                                               ; preds = %23, %26, %27
  %32 = phi i8 [ 111, %23 ], [ 100, %26 ], [ %30, %27 ]
  %33 = getelementptr inbounds nuw i8, ptr %24, i64 1
  %34 = getelementptr inbounds nuw i8, ptr %24, i64 2
  store i8 %32, ptr %33, align 1
  store i8 0, ptr %34, align 1
  %35 = call i32 (ptr, i64, ptr, ...) @sprintf_s(ptr noundef nonnull %7, i64 noundef 64, ptr noundef nonnull %8, i32 noundef %5)
  %36 = sext i32 %35 to i64
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %9, ptr noundef nonnull align 8 dereferenceable(16) %2, i64 16, i1 false)
  call void @"?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr dead_on_unwind writable sret(%"class.std::ostreambuf_iterator") align 8 %1, ptr dead_on_return noundef nonnull %9, ptr noundef nonnull align 8 dereferenceable(72) %3, i8 noundef %4, ptr noundef nonnull %7, i64 noundef %36)
  call void @llvm.lifetime.end.p0(i64 6, ptr nonnull %8) #10
  call void @llvm.lifetime.end.p0(i64 64, ptr nonnull %7) #10
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_N@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr dead_on_unwind noalias writable sret(%"class.std::ostreambuf_iterator") align 8 %1, ptr dead_on_return noundef %2, ptr noundef nonnull align 8 dereferenceable(72) %3, i8 noundef %4, i1 noundef zeroext %5) unnamed_addr #2 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %7 = alloca %"class.std::ostreambuf_iterator", align 8
  %8 = alloca %"class.std::locale", align 8
  %9 = alloca %"class.std::basic_string", align 8
  %10 = alloca %"class.std::basic_string", align 8
  %11 = alloca %"class.std::basic_string", align 8
  %12 = alloca [7 x i8], align 1
  %13 = alloca [7 x i8], align 1
  %14 = alloca [7 x i8], align 1
  %15 = getelementptr inbounds nuw i8, ptr %3, i64 24
  %16 = load i32, ptr %15, align 8
  %17 = and i32 %16, 16384
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %19, label %24

19:                                               ; preds = %6
  %20 = zext i1 %5 to i32
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %7, ptr noundef nonnull align 8 dereferenceable(16) %2, i64 16, i1 false)
  %21 = load ptr, ptr %0, align 8
  %22 = getelementptr inbounds nuw i8, ptr %21, i64 72
  %23 = load ptr, ptr %22, align 8
  call void %23(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr dead_on_unwind writable sret(%"class.std::ostreambuf_iterator") align 8 %1, ptr dead_on_return noundef nonnull %7, ptr noundef nonnull align 8 dereferenceable(72) %3, i8 noundef %4, i32 noundef %20)
  br label %297

24:                                               ; preds = %6
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %8) #10
  tail call void @llvm.experimental.noalias.scope.decl(metadata !42)
  %25 = getelementptr inbounds nuw i8, ptr %3, i64 64
  %26 = load ptr, ptr %25, align 8, !noalias !42
  %27 = getelementptr inbounds nuw i8, ptr %8, i64 8
  %28 = getelementptr inbounds nuw i8, ptr %26, i64 8
  %29 = load ptr, ptr %28, align 8, !noalias !42
  store ptr %29, ptr %27, align 8, !alias.scope !42
  %30 = load ptr, ptr %29, align 8, !noalias !42
  %31 = getelementptr inbounds nuw i8, ptr %30, i64 8
  %32 = load ptr, ptr %31, align 8, !noalias !42
  tail call void %32(ptr noundef nonnull align 8 dereferenceable(16) %29) #10, !noalias !42
  %33 = invoke noundef nonnull align 8 dereferenceable(48) ptr @"??$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z"(ptr noundef nonnull align 8 dereferenceable(16) %8)
          to label %34 unwind label %81

34:                                               ; preds = %24
  %35 = load ptr, ptr %27, align 8
  %36 = icmp eq ptr %35, null
  br i1 %36, label %47, label %37

37:                                               ; preds = %34
  %38 = load ptr, ptr %35, align 8
  %39 = getelementptr inbounds nuw i8, ptr %38, i64 16
  %40 = load ptr, ptr %39, align 8
  %41 = call noundef ptr %40(ptr noundef nonnull align 8 dereferenceable(16) %35) #10
  %42 = icmp eq ptr %41, null
  br i1 %42, label %47, label %43

43:                                               ; preds = %37
  %44 = load ptr, ptr %41, align 8
  %45 = load ptr, ptr %44, align 8
  %46 = call noundef ptr %45(ptr noundef nonnull align 8 dereferenceable(8) %41, i32 noundef 1) #10
  br label %47

47:                                               ; preds = %34, %37, %43
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %8) #10
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %9) #10
  %48 = getelementptr inbounds nuw i8, ptr %9, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %9, i8 0, i64 32, i1 false)
  store i64 15, ptr %48, align 8
  store i8 0, ptr %9, align 8
  br i1 %5, label %49, label %96

49:                                               ; preds = %47
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %10) #10
  %50 = load ptr, ptr %33, align 8, !noalias !45
  %51 = getelementptr inbounds nuw i8, ptr %50, i64 56
  %52 = load ptr, ptr %51, align 8, !noalias !45
  invoke void %52(ptr noundef nonnull align 8 dereferenceable(48) %33, ptr dead_on_unwind nonnull writable sret(%"class.std::basic_string") align 8 %10)
          to label %53 unwind label %295

53:                                               ; preds = %49
  %54 = load i64, ptr %48, align 8
  %55 = icmp ugt i64 %54, 15
  br i1 %55, label %56, label %77

56:                                               ; preds = %53
  %57 = load ptr, ptr %9, align 8
  %58 = add i64 %54, 1
  %59 = icmp ugt i64 %58, 4095
  br i1 %59, label %60, label %74

60:                                               ; preds = %56
  %61 = getelementptr inbounds i8, ptr %57, i64 -8
  %62 = load i64, ptr %61, align 8
  %63 = ptrtoint ptr %57 to i64
  %64 = add i64 %63, -8
  %65 = sub i64 %64, %62
  %66 = icmp ult i64 %65, 32
  br i1 %66, label %69, label %67

67:                                               ; preds = %60
  invoke void @_invalid_parameter_noinfo_noreturn() #29
          to label %68 unwind label %72

68:                                               ; preds = %67
  unreachable

69:                                               ; preds = %60
  %70 = add i64 %54, 40
  %71 = inttoptr i64 %62 to ptr
  br label %74

72:                                               ; preds = %67
  %73 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %73) ]
  unreachable

74:                                               ; preds = %69, %56
  %75 = phi i64 [ %70, %69 ], [ %58, %56 ]
  %76 = phi ptr [ %71, %69 ], [ %57, %56 ]
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %76, i64 noundef %75) #10
  br label %77

77:                                               ; preds = %53, %74
  %78 = getelementptr inbounds nuw i8, ptr %9, i64 16
  store i64 0, ptr %78, align 8
  store i64 15, ptr %48, align 8
  store i8 0, ptr %9, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %9, ptr noundef nonnull align 8 dereferenceable(32) %10, i64 32, i1 false)
  %79 = getelementptr inbounds nuw i8, ptr %10, i64 16
  store i64 0, ptr %79, align 8
  %80 = getelementptr inbounds nuw i8, ptr %10, i64 24
  store i64 15, ptr %80, align 8
  store i8 0, ptr %10, align 8
  store i64 0, ptr %79, align 8
  store i64 15, ptr %80, align 8
  store i8 0, ptr %10, align 8
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %10) #10
  br label %128

81:                                               ; preds = %24
  %82 = cleanuppad within none []
  %83 = load ptr, ptr %27, align 8
  %84 = icmp eq ptr %83, null
  br i1 %84, label %95, label %85

85:                                               ; preds = %81
  %86 = load ptr, ptr %83, align 8
  %87 = getelementptr inbounds nuw i8, ptr %86, i64 16
  %88 = load ptr, ptr %87, align 8
  %89 = call noundef ptr %88(ptr noundef nonnull align 8 dereferenceable(16) %83) #10 [ "funclet"(token %82) ]
  %90 = icmp eq ptr %89, null
  br i1 %90, label %95, label %91

91:                                               ; preds = %85
  %92 = load ptr, ptr %89, align 8
  %93 = load ptr, ptr %92, align 8
  %94 = call noundef ptr %93(ptr noundef nonnull align 8 dereferenceable(8) %89, i32 noundef 1) #10 [ "funclet"(token %82) ]
  br label %95

95:                                               ; preds = %81, %85, %91
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %8) #10
  cleanupret from %82 unwind to caller

96:                                               ; preds = %47
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %11) #10
  %97 = load ptr, ptr %33, align 8, !noalias !48
  %98 = getelementptr inbounds nuw i8, ptr %97, i64 48
  %99 = load ptr, ptr %98, align 8, !noalias !48
  invoke void %99(ptr noundef nonnull align 8 dereferenceable(48) %33, ptr dead_on_unwind nonnull writable sret(%"class.std::basic_string") align 8 %11)
          to label %100 unwind label %295

100:                                              ; preds = %96
  %101 = load i64, ptr %48, align 8
  %102 = icmp ugt i64 %101, 15
  br i1 %102, label %103, label %124

103:                                              ; preds = %100
  %104 = load ptr, ptr %9, align 8
  %105 = add i64 %101, 1
  %106 = icmp ugt i64 %105, 4095
  br i1 %106, label %107, label %121

107:                                              ; preds = %103
  %108 = getelementptr inbounds i8, ptr %104, i64 -8
  %109 = load i64, ptr %108, align 8
  %110 = ptrtoint ptr %104 to i64
  %111 = add i64 %110, -8
  %112 = sub i64 %111, %109
  %113 = icmp ult i64 %112, 32
  br i1 %113, label %116, label %114

114:                                              ; preds = %107
  invoke void @_invalid_parameter_noinfo_noreturn() #29
          to label %115 unwind label %119

115:                                              ; preds = %114
  unreachable

116:                                              ; preds = %107
  %117 = add i64 %101, 40
  %118 = inttoptr i64 %109 to ptr
  br label %121

119:                                              ; preds = %114
  %120 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %120) ]
  unreachable

121:                                              ; preds = %116, %103
  %122 = phi i64 [ %117, %116 ], [ %105, %103 ]
  %123 = phi ptr [ %118, %116 ], [ %104, %103 ]
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %123, i64 noundef %122) #10
  br label %124

124:                                              ; preds = %100, %121
  %125 = getelementptr inbounds nuw i8, ptr %9, i64 16
  store i64 0, ptr %125, align 8
  store i64 15, ptr %48, align 8
  store i8 0, ptr %9, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %9, ptr noundef nonnull align 8 dereferenceable(32) %11, i64 32, i1 false)
  %126 = getelementptr inbounds nuw i8, ptr %11, i64 16
  store i64 0, ptr %126, align 8
  %127 = getelementptr inbounds nuw i8, ptr %11, i64 24
  store i64 15, ptr %127, align 8
  store i8 0, ptr %11, align 8
  store i64 0, ptr %126, align 8
  store i64 15, ptr %127, align 8
  store i8 0, ptr %11, align 8
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %11) #10
  br label %128

128:                                              ; preds = %124, %77
  %129 = getelementptr inbounds nuw i8, ptr %3, i64 40
  %130 = load i64, ptr %129, align 8
  %131 = getelementptr inbounds nuw i8, ptr %9, i64 16
  %132 = load i64, ptr %131, align 8
  %133 = call i64 @llvm.usub.sat.i64(i64 %130, i64 %132)
  %134 = icmp sgt i64 %130, 0
  %135 = select i1 %134, i64 %133, i64 0
  %136 = load i32, ptr %15, align 8
  %137 = and i32 %136, 448
  %138 = icmp eq i32 %137, 64
  br i1 %138, label %180, label %139

139:                                              ; preds = %128
  %140 = load i8, ptr %2, align 8
  %141 = getelementptr inbounds nuw i8, ptr %2, i64 1
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %12, ptr noundef nonnull align 1 dereferenceable(7) %141, i64 7, i1 false)
  %142 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %143 = load ptr, ptr %142, align 8
  %144 = icmp eq i64 %135, 0
  br i1 %144, label %178, label %145

145:                                              ; preds = %139
  %146 = zext i8 %4 to i32
  %147 = icmp eq ptr %143, null
  %148 = getelementptr inbounds nuw i8, ptr %143, i64 64
  %149 = getelementptr inbounds nuw i8, ptr %143, i64 88
  br label %150

150:                                              ; preds = %174, %145
  %151 = phi i8 [ %140, %145 ], [ %175, %174 ]
  %152 = phi i64 [ %135, %145 ], [ %176, %174 ]
  br i1 %147, label %173, label %153

153:                                              ; preds = %150
  %154 = load ptr, ptr %148, align 8, !noalias !51
  %155 = load ptr, ptr %154, align 8, !noalias !51
  %156 = icmp eq ptr %155, null
  br i1 %156, label %166, label %157

157:                                              ; preds = %153
  %158 = load ptr, ptr %149, align 8, !noalias !51
  %159 = load i32, ptr %158, align 4, !noalias !51
  %160 = icmp sgt i32 %159, 0
  br i1 %160, label %161, label %166

161:                                              ; preds = %157
  %162 = add nsw i32 %159, -1
  store i32 %162, ptr %158, align 4, !noalias !51
  %163 = load ptr, ptr %148, align 8, !noalias !51
  %164 = load ptr, ptr %163, align 8, !noalias !51
  %165 = getelementptr inbounds nuw i8, ptr %164, i64 1
  store ptr %165, ptr %163, align 8, !noalias !51
  store i8 %4, ptr %164, align 1, !noalias !51
  br label %174

166:                                              ; preds = %157, %153
  %167 = load ptr, ptr %143, align 8, !noalias !51
  %168 = getelementptr inbounds nuw i8, ptr %167, i64 24
  %169 = load ptr, ptr %168, align 8, !noalias !51
  %170 = invoke noundef i32 %169(ptr noundef nonnull align 8 dereferenceable(104) %143, i32 noundef %146)
          to label %171 unwind label %295

171:                                              ; preds = %166
  %172 = icmp eq i32 %170, -1
  br i1 %172, label %173, label %174

173:                                              ; preds = %171, %150
  br label %174

174:                                              ; preds = %173, %171, %161
  %175 = phi i8 [ 1, %173 ], [ %151, %171 ], [ %151, %161 ]
  %176 = add i64 %152, -1
  %177 = icmp eq i64 %176, 0
  br i1 %177, label %178, label %150, !llvm.loop !54

178:                                              ; preds = %174, %139
  %179 = phi i8 [ %140, %139 ], [ %175, %174 ]
  store i8 %179, ptr %2, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %141, ptr noundef nonnull align 1 dereferenceable(7) %12, i64 7, i1 false)
  store ptr %143, ptr %142, align 8
  br label %180

180:                                              ; preds = %178, %128
  %181 = phi i64 [ 0, %178 ], [ %135, %128 ]
  %182 = getelementptr inbounds nuw i8, ptr %9, i64 16
  %183 = load i64, ptr %182, align 8
  %184 = load i64, ptr %48, align 8
  %185 = load ptr, ptr %9, align 8
  %186 = load i8, ptr %2, align 8
  %187 = getelementptr inbounds nuw i8, ptr %2, i64 1
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %13, ptr noundef nonnull align 1 dereferenceable(7) %187, i64 7, i1 false)
  %188 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %189 = load ptr, ptr %188, align 8
  %190 = icmp eq i64 %183, 0
  br i1 %190, label %229, label %191

191:                                              ; preds = %180
  %192 = icmp ugt i64 %184, 15
  %193 = select i1 %192, ptr %185, ptr %9
  %194 = icmp eq ptr %189, null
  %195 = getelementptr inbounds nuw i8, ptr %189, i64 64
  %196 = getelementptr inbounds nuw i8, ptr %189, i64 88
  br label %197

197:                                              ; preds = %191, %224
  %198 = phi i8 [ %225, %224 ], [ %186, %191 ]
  %199 = phi i64 [ %226, %224 ], [ %183, %191 ]
  %200 = phi ptr [ %227, %224 ], [ %193, %191 ]
  %201 = load i8, ptr %200, align 1, !noalias !55
  br i1 %194, label %223, label %202

202:                                              ; preds = %197
  %203 = load ptr, ptr %195, align 8, !noalias !55
  %204 = load ptr, ptr %203, align 8, !noalias !55
  %205 = icmp eq ptr %204, null
  br i1 %205, label %215, label %206

206:                                              ; preds = %202
  %207 = load ptr, ptr %196, align 8, !noalias !55
  %208 = load i32, ptr %207, align 4, !noalias !55
  %209 = icmp sgt i32 %208, 0
  br i1 %209, label %210, label %215

210:                                              ; preds = %206
  %211 = add nsw i32 %208, -1
  store i32 %211, ptr %207, align 4, !noalias !55
  %212 = load ptr, ptr %195, align 8, !noalias !55
  %213 = load ptr, ptr %212, align 8, !noalias !55
  %214 = getelementptr inbounds nuw i8, ptr %213, i64 1
  store ptr %214, ptr %212, align 8, !noalias !55
  store i8 %201, ptr %213, align 1, !noalias !55
  br label %224

215:                                              ; preds = %206, %202
  %216 = zext i8 %201 to i32
  %217 = load ptr, ptr %189, align 8, !noalias !55
  %218 = getelementptr inbounds nuw i8, ptr %217, i64 24
  %219 = load ptr, ptr %218, align 8, !noalias !55
  %220 = invoke noundef i32 %219(ptr noundef nonnull align 8 dereferenceable(104) %189, i32 noundef %216)
          to label %221 unwind label %295

221:                                              ; preds = %215
  %222 = icmp eq i32 %220, -1
  br i1 %222, label %223, label %224

223:                                              ; preds = %221, %197
  br label %224

224:                                              ; preds = %223, %221, %210
  %225 = phi i8 [ 1, %223 ], [ %198, %221 ], [ %198, %210 ]
  %226 = add i64 %199, -1
  %227 = getelementptr inbounds nuw i8, ptr %200, i64 1
  %228 = icmp eq i64 %226, 0
  br i1 %228, label %229, label %197, !llvm.loop !58

229:                                              ; preds = %224, %180
  %230 = phi i8 [ %186, %180 ], [ %225, %224 ]
  store i8 %230, ptr %2, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %187, ptr noundef nonnull align 1 dereferenceable(7) %13, i64 7, i1 false)
  store ptr %189, ptr %188, align 8
  store i64 0, ptr %129, align 8
  %231 = load i8, ptr %2, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %14, ptr noundef nonnull align 1 dereferenceable(7) %187, i64 7, i1 false)
  %232 = load ptr, ptr %188, align 8
  %233 = icmp eq i64 %181, 0
  br i1 %233, label %267, label %234

234:                                              ; preds = %229
  %235 = zext i8 %4 to i32
  %236 = icmp eq ptr %232, null
  %237 = getelementptr inbounds nuw i8, ptr %232, i64 64
  %238 = getelementptr inbounds nuw i8, ptr %232, i64 88
  br label %239

239:                                              ; preds = %263, %234
  %240 = phi i8 [ %231, %234 ], [ %264, %263 ]
  %241 = phi i64 [ %181, %234 ], [ %265, %263 ]
  br i1 %236, label %262, label %242

242:                                              ; preds = %239
  %243 = load ptr, ptr %237, align 8, !noalias !59
  %244 = load ptr, ptr %243, align 8, !noalias !59
  %245 = icmp eq ptr %244, null
  br i1 %245, label %255, label %246

246:                                              ; preds = %242
  %247 = load ptr, ptr %238, align 8, !noalias !59
  %248 = load i32, ptr %247, align 4, !noalias !59
  %249 = icmp sgt i32 %248, 0
  br i1 %249, label %250, label %255

250:                                              ; preds = %246
  %251 = add nsw i32 %248, -1
  store i32 %251, ptr %247, align 4, !noalias !59
  %252 = load ptr, ptr %237, align 8, !noalias !59
  %253 = load ptr, ptr %252, align 8, !noalias !59
  %254 = getelementptr inbounds nuw i8, ptr %253, i64 1
  store ptr %254, ptr %252, align 8, !noalias !59
  store i8 %4, ptr %253, align 1, !noalias !59
  br label %263

255:                                              ; preds = %246, %242
  %256 = load ptr, ptr %232, align 8, !noalias !59
  %257 = getelementptr inbounds nuw i8, ptr %256, i64 24
  %258 = load ptr, ptr %257, align 8, !noalias !59
  %259 = invoke noundef i32 %258(ptr noundef nonnull align 8 dereferenceable(104) %232, i32 noundef %235)
          to label %260 unwind label %295

260:                                              ; preds = %255
  %261 = icmp eq i32 %259, -1
  br i1 %261, label %262, label %263

262:                                              ; preds = %260, %239
  br label %263

263:                                              ; preds = %262, %260, %250
  %264 = phi i8 [ 1, %262 ], [ %240, %260 ], [ %240, %250 ]
  %265 = add i64 %241, -1
  %266 = icmp eq i64 %265, 0
  br i1 %266, label %267, label %239, !llvm.loop !54

267:                                              ; preds = %263, %229
  %268 = phi i8 [ %231, %229 ], [ %264, %263 ]
  store i8 %268, ptr %1, align 8
  %269 = getelementptr inbounds nuw i8, ptr %1, i64 1
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %269, ptr noundef nonnull align 1 dereferenceable(7) %14, i64 7, i1 false)
  %270 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store ptr %232, ptr %270, align 8
  %271 = load i64, ptr %48, align 8
  %272 = icmp ugt i64 %271, 15
  br i1 %272, label %273, label %294

273:                                              ; preds = %267
  %274 = load ptr, ptr %9, align 8
  %275 = add i64 %271, 1
  %276 = icmp ugt i64 %275, 4095
  br i1 %276, label %277, label %291

277:                                              ; preds = %273
  %278 = getelementptr inbounds i8, ptr %274, i64 -8
  %279 = load i64, ptr %278, align 8
  %280 = ptrtoint ptr %274 to i64
  %281 = add i64 %280, -8
  %282 = sub i64 %281, %279
  %283 = icmp ult i64 %282, 32
  br i1 %283, label %286, label %284

284:                                              ; preds = %277
  invoke void @_invalid_parameter_noinfo_noreturn() #29
          to label %285 unwind label %289

285:                                              ; preds = %284
  unreachable

286:                                              ; preds = %277
  %287 = add i64 %271, 40
  %288 = inttoptr i64 %279 to ptr
  br label %291

289:                                              ; preds = %284
  %290 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %290) ]
  unreachable

291:                                              ; preds = %286, %273
  %292 = phi i64 [ %287, %286 ], [ %275, %273 ]
  %293 = phi ptr [ %288, %286 ], [ %274, %273 ]
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %293, i64 noundef %292) #10
  br label %294

294:                                              ; preds = %267, %291
  store i64 0, ptr %182, align 8
  store i64 15, ptr %48, align 8
  store i8 0, ptr %9, align 8
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %9) #10
  br label %297

295:                                              ; preds = %255, %215, %166, %96, %49
  %296 = cleanuppad within none []
  call void @"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %9) #10 [ "funclet"(token %296) ]
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %9) #10
  cleanupret from %296 unwind to caller

297:                                              ; preds = %294, %19
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @"?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr dead_on_unwind noalias writable sret(%"class.std::ostreambuf_iterator") align 8 %1, ptr dead_on_return noundef %2, ptr noundef nonnull align 8 dereferenceable(72) %3, i8 noundef %4, ptr noundef %5, i64 noundef %6) local_unnamed_addr #2 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %8 = alloca %"class.std::locale", align 8
  %9 = alloca %"class.std::basic_string", align 8
  %10 = alloca %"class.std::locale", align 8
  %11 = alloca %"class.std::basic_string", align 8
  %12 = alloca [7 x i8], align 1
  %13 = alloca [7 x i8], align 1
  %14 = alloca [7 x i8], align 1
  %15 = alloca [7 x i8], align 1
  %16 = alloca [7 x i8], align 1
  %17 = icmp eq i64 %6, 0
  br i1 %17, label %24, label %18

18:                                               ; preds = %7
  %19 = load i8, ptr %5, align 1
  %20 = icmp eq i8 %19, 43
  br i1 %20, label %24, label %21

21:                                               ; preds = %18
  %22 = icmp eq i8 %19, 45
  %23 = zext i1 %22 to i64
  br label %24

24:                                               ; preds = %18, %21, %7
  %25 = phi i64 [ 0, %7 ], [ 1, %18 ], [ %23, %21 ]
  %26 = getelementptr inbounds nuw i8, ptr %3, i64 24
  %27 = load i32, ptr %26, align 8
  %28 = and i32 %27, 3584
  %29 = icmp eq i32 %28, 2048
  br i1 %29, label %30, label %41

30:                                               ; preds = %24
  %31 = or disjoint i64 %25, 2
  %32 = icmp ugt i64 %31, %6
  br i1 %32, label %41, label %33

33:                                               ; preds = %30
  %34 = getelementptr inbounds nuw i8, ptr %5, i64 %25
  %35 = load i8, ptr %34, align 1
  %36 = icmp eq i8 %35, 48
  br i1 %36, label %37, label %41

37:                                               ; preds = %33
  %38 = getelementptr inbounds nuw i8, ptr %34, i64 1
  %39 = load i8, ptr %38, align 1
  switch i8 %39, label %41 [
    i8 120, label %40
    i8 88, label %40
  ]

40:                                               ; preds = %37, %37
  br label %41

41:                                               ; preds = %37, %40, %33, %30, %24
  %42 = phi i64 [ %31, %40 ], [ %25, %33 ], [ %25, %30 ], [ %25, %24 ], [ %25, %37 ]
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %8) #10
  tail call void @llvm.experimental.noalias.scope.decl(metadata !62)
  %43 = getelementptr inbounds nuw i8, ptr %3, i64 64
  %44 = load ptr, ptr %43, align 8, !noalias !62
  %45 = getelementptr inbounds nuw i8, ptr %8, i64 8
  %46 = getelementptr inbounds nuw i8, ptr %44, i64 8
  %47 = load ptr, ptr %46, align 8, !noalias !62
  store ptr %47, ptr %45, align 8, !alias.scope !62
  %48 = load ptr, ptr %47, align 8, !noalias !62
  %49 = getelementptr inbounds nuw i8, ptr %48, i64 8
  %50 = load ptr, ptr %49, align 8, !noalias !62
  tail call void %50(ptr noundef nonnull align 8 dereferenceable(16) %47) #10, !noalias !62
  %51 = invoke noundef nonnull align 8 dereferenceable(48) ptr @"??$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z"(ptr noundef nonnull align 8 dereferenceable(16) %8)
          to label %52 unwind label %185

52:                                               ; preds = %41
  %53 = load ptr, ptr %45, align 8
  %54 = icmp eq ptr %53, null
  br i1 %54, label %65, label %55

55:                                               ; preds = %52
  %56 = load ptr, ptr %53, align 8
  %57 = getelementptr inbounds nuw i8, ptr %56, i64 16
  %58 = load ptr, ptr %57, align 8
  %59 = call noundef ptr %58(ptr noundef nonnull align 8 dereferenceable(16) %53) #10
  %60 = icmp eq ptr %59, null
  br i1 %60, label %65, label %61

61:                                               ; preds = %55
  %62 = load ptr, ptr %59, align 8
  %63 = load ptr, ptr %62, align 8
  %64 = call noundef ptr %63(ptr noundef nonnull align 8 dereferenceable(8) %59, i32 noundef 1) #10
  br label %65

65:                                               ; preds = %52, %55, %61
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %8) #10
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %9) #10
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %9, i8 0, i64 32, i1 false)
  %66 = icmp slt i64 %6, 0
  br i1 %66, label %67, label %68

67:                                               ; preds = %65
  call void @"?_Xlen_string@std@@YAXXZ"() #29
  unreachable

68:                                               ; preds = %65
  %69 = icmp ult i64 %6, 16
  br i1 %69, label %70, label %74

70:                                               ; preds = %68
  %71 = getelementptr inbounds nuw i8, ptr %9, i64 16
  store i64 %6, ptr %71, align 8
  %72 = getelementptr inbounds nuw i8, ptr %9, i64 24
  store i64 15, ptr %72, align 8
  call void @llvm.memset.p0.i64(ptr nonnull align 8 dereferenceable(32) %9, i8 0, i64 %6, i1 false)
  %73 = getelementptr inbounds nuw [16 x i8], ptr %9, i64 0, i64 %6
  store i8 0, ptr %73, align 1
  br label %97

74:                                               ; preds = %68
  %75 = or i64 %6, 15
  %76 = call i64 @llvm.umax.i64(i64 %75, i64 22)
  %77 = icmp ugt i64 %75, 4094
  br i1 %77, label %78, label %89

78:                                               ; preds = %74
  %79 = icmp ult i64 %75, -40
  br i1 %79, label %81, label %80

80:                                               ; preds = %78
  call void @"?_Throw_bad_array_new_length@std@@YAXXZ"() #29
  unreachable

81:                                               ; preds = %78
  %82 = add nuw i64 %76, 40
  %83 = call noalias noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef %82) #31
  %84 = ptrtoint ptr %83 to i64
  %85 = add i64 %84, 39
  %86 = and i64 %85, -32
  %87 = inttoptr i64 %86 to ptr
  %88 = getelementptr inbounds i8, ptr %87, i64 -8
  store i64 %84, ptr %88, align 8
  br label %92

89:                                               ; preds = %74
  %90 = add nuw nsw i64 %76, 1
  %91 = call noalias noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef %90) #31
  br label %92

92:                                               ; preds = %89, %81
  %93 = phi ptr [ %87, %81 ], [ %91, %89 ]
  store ptr %93, ptr %9, align 8
  %94 = getelementptr inbounds nuw i8, ptr %9, i64 16
  store i64 %6, ptr %94, align 8
  %95 = getelementptr inbounds nuw i8, ptr %9, i64 24
  store i64 %76, ptr %95, align 8
  call void @llvm.memset.p0.i64(ptr align 1 %93, i8 0, i64 %6, i1 false)
  %96 = getelementptr inbounds nuw i8, ptr %93, i64 %6
  store i8 0, ptr %96, align 1
  br label %97

97:                                               ; preds = %70, %92
  %98 = getelementptr inbounds nuw i8, ptr %9, i64 24
  %99 = load i64, ptr %98, align 8
  %100 = icmp ugt i64 %99, 15
  %101 = load ptr, ptr %9, align 8
  %102 = select i1 %100, ptr %101, ptr %9
  %103 = getelementptr inbounds nuw i8, ptr %5, i64 %6
  %104 = load ptr, ptr %51, align 8
  %105 = getelementptr inbounds nuw i8, ptr %104, i64 56
  %106 = load ptr, ptr %105, align 8
  %107 = invoke noundef ptr %106(ptr noundef nonnull align 8 dereferenceable(48) %51, ptr noundef %5, ptr noundef %103, ptr noundef nonnull %102)
          to label %108 unwind label %585

108:                                              ; preds = %97
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %10) #10
  call void @llvm.experimental.noalias.scope.decl(metadata !65)
  %109 = load ptr, ptr %43, align 8, !noalias !65
  %110 = getelementptr inbounds nuw i8, ptr %10, i64 8
  %111 = getelementptr inbounds nuw i8, ptr %109, i64 8
  %112 = load ptr, ptr %111, align 8, !noalias !65
  store ptr %112, ptr %110, align 8, !alias.scope !65
  %113 = load ptr, ptr %112, align 8, !noalias !65
  %114 = getelementptr inbounds nuw i8, ptr %113, i64 8
  %115 = load ptr, ptr %114, align 8, !noalias !65
  call void %115(ptr noundef nonnull align 8 dereferenceable(16) %112) #10, !noalias !65
  %116 = invoke noundef nonnull align 8 dereferenceable(48) ptr @"??$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z"(ptr noundef nonnull align 8 dereferenceable(16) %10)
          to label %117 unwind label %200

117:                                              ; preds = %108
  %118 = load ptr, ptr %110, align 8
  %119 = icmp eq ptr %118, null
  br i1 %119, label %130, label %120

120:                                              ; preds = %117
  %121 = load ptr, ptr %118, align 8
  %122 = getelementptr inbounds nuw i8, ptr %121, i64 16
  %123 = load ptr, ptr %122, align 8
  %124 = call noundef ptr %123(ptr noundef nonnull align 8 dereferenceable(16) %118) #10
  %125 = icmp eq ptr %124, null
  br i1 %125, label %130, label %126

126:                                              ; preds = %120
  %127 = load ptr, ptr %124, align 8
  %128 = load ptr, ptr %127, align 8
  %129 = call noundef ptr %128(ptr noundef nonnull align 8 dereferenceable(8) %124, i32 noundef 1) #10
  br label %130

130:                                              ; preds = %117, %120, %126
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %10) #10
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %11) #10
  %131 = load ptr, ptr %116, align 8, !noalias !68
  %132 = getelementptr inbounds nuw i8, ptr %131, i64 40
  %133 = load ptr, ptr %132, align 8, !noalias !68
  invoke void %133(ptr noundef nonnull align 8 dereferenceable(48) %116, ptr dead_on_unwind nonnull writable sret(%"class.std::basic_string") align 8 %11)
          to label %134 unwind label %585

134:                                              ; preds = %130
  %135 = getelementptr inbounds nuw i8, ptr %11, i64 24
  %136 = load i64, ptr %135, align 8
  %137 = icmp ugt i64 %136, 15
  %138 = load ptr, ptr %11, align 8
  %139 = select i1 %137, ptr %138, ptr %11
  %140 = load i8, ptr %139, align 1
  %141 = add i8 %140, -1
  %142 = icmp ult i8 %141, 126
  br i1 %142, label %143, label %215

143:                                              ; preds = %134
  %144 = load ptr, ptr %116, align 8
  %145 = getelementptr inbounds nuw i8, ptr %144, i64 32
  %146 = load ptr, ptr %145, align 8
  %147 = invoke noundef i8 %146(ptr noundef nonnull align 8 dereferenceable(48) %116)
          to label %148 unwind label %583

148:                                              ; preds = %143
  %149 = getelementptr inbounds nuw i8, ptr %9, i64 16
  br label %150

150:                                              ; preds = %148, %180
  %151 = phi ptr [ %184, %180 ], [ %139, %148 ]
  %152 = phi i64 [ %161, %180 ], [ %6, %148 ]
  %153 = load i8, ptr %151, align 1
  %154 = add i8 %153, -1
  %155 = icmp ult i8 %154, 126
  br i1 %155, label %156, label %215

156:                                              ; preds = %150
  %157 = zext nneg i8 %153 to i64
  %158 = sub i64 %152, %42
  %159 = icmp ugt i64 %158, %157
  br i1 %159, label %160, label %215

160:                                              ; preds = %156
  %161 = sub i64 %152, %157
  %162 = load i64, ptr %149, align 8
  %163 = icmp ult i64 %162, %161
  br i1 %163, label %164, label %166

164:                                              ; preds = %160
  invoke void @"?_Xran@?$_String_val@U?$_Simple_types@D@std@@@std@@SAXXZ"() #29
          to label %165 unwind label %583

165:                                              ; preds = %164
  unreachable

166:                                              ; preds = %160
  %167 = load i64, ptr %98, align 8
  %168 = icmp eq i64 %167, %162
  br i1 %168, label %178, label %169

169:                                              ; preds = %166
  %170 = add i64 %162, 1
  store i64 %170, ptr %149, align 8
  %171 = icmp ugt i64 %167, 15
  %172 = load ptr, ptr %9, align 8
  %173 = select i1 %171, ptr %172, ptr %9
  %174 = getelementptr inbounds nuw i8, ptr %173, i64 %161
  %175 = sub i64 %162, %161
  %176 = add i64 %175, 1
  %177 = getelementptr inbounds nuw i8, ptr %174, i64 1
  call void @llvm.memmove.p0.p0.i64(ptr nonnull align 1 %177, ptr align 1 %174, i64 %176, i1 false)
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(1) %174, i8 %147, i64 1, i1 false)
  br label %180

178:                                              ; preds = %166
  %179 = invoke noundef nonnull align 8 dereferenceable(32) ptr @"??$_Reallocate_grow_by@V<lambda_1>@?0??insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@_K0D@Z@_K_KD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??insert@01@QEAAAEAV01@00D@Z@_K2D@Z"(ptr noundef nonnull align 8 dereferenceable(32) %9, i64 noundef 1, i8 undef, i64 noundef %161, i64 noundef 1, i8 noundef %147)
          to label %180 unwind label %583

180:                                              ; preds = %169, %178
  %181 = getelementptr inbounds nuw i8, ptr %151, i64 1
  %182 = load i8, ptr %181, align 1
  %183 = icmp sgt i8 %182, 0
  %184 = select i1 %183, ptr %181, ptr %151
  br label %150, !llvm.loop !71

185:                                              ; preds = %41
  %186 = cleanuppad within none []
  %187 = load ptr, ptr %45, align 8
  %188 = icmp eq ptr %187, null
  br i1 %188, label %199, label %189

189:                                              ; preds = %185
  %190 = load ptr, ptr %187, align 8
  %191 = getelementptr inbounds nuw i8, ptr %190, i64 16
  %192 = load ptr, ptr %191, align 8
  %193 = call noundef ptr %192(ptr noundef nonnull align 8 dereferenceable(16) %187) #10 [ "funclet"(token %186) ]
  %194 = icmp eq ptr %193, null
  br i1 %194, label %199, label %195

195:                                              ; preds = %189
  %196 = load ptr, ptr %193, align 8
  %197 = load ptr, ptr %196, align 8
  %198 = call noundef ptr %197(ptr noundef nonnull align 8 dereferenceable(8) %193, i32 noundef 1) #10 [ "funclet"(token %186) ]
  br label %199

199:                                              ; preds = %185, %189, %195
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %8) #10
  cleanupret from %186 unwind to caller

200:                                              ; preds = %108
  %201 = cleanuppad within none []
  %202 = load ptr, ptr %110, align 8
  %203 = icmp eq ptr %202, null
  br i1 %203, label %214, label %204

204:                                              ; preds = %200
  %205 = load ptr, ptr %202, align 8
  %206 = getelementptr inbounds nuw i8, ptr %205, i64 16
  %207 = load ptr, ptr %206, align 8
  %208 = call noundef ptr %207(ptr noundef nonnull align 8 dereferenceable(16) %202) #10 [ "funclet"(token %201) ]
  %209 = icmp eq ptr %208, null
  br i1 %209, label %214, label %210

210:                                              ; preds = %204
  %211 = load ptr, ptr %208, align 8
  %212 = load ptr, ptr %211, align 8
  %213 = call noundef ptr %212(ptr noundef nonnull align 8 dereferenceable(8) %208, i32 noundef 1) #10 [ "funclet"(token %201) ]
  br label %214

214:                                              ; preds = %200, %204, %210
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %10) #10
  cleanupret from %201 unwind label %585

215:                                              ; preds = %156, %150, %134
  %216 = getelementptr inbounds nuw i8, ptr %9, i64 16
  %217 = load i64, ptr %216, align 8
  %218 = getelementptr inbounds nuw i8, ptr %3, i64 40
  %219 = load i64, ptr %218, align 8
  %220 = icmp sgt i64 %219, 0
  %221 = icmp ugt i64 %219, %217
  %222 = select i1 %220, i1 %221, i1 false
  %223 = sub i64 %219, %217
  %224 = select i1 %222, i64 %223, i64 0
  %225 = load i32, ptr %26, align 8
  %226 = and i32 %225, 448
  switch i32 %226, label %227 [
    i32 256, label %311
    i32 64, label %395
  ]

227:                                              ; preds = %215
  %228 = load i8, ptr %2, align 8
  %229 = getelementptr inbounds nuw i8, ptr %2, i64 1
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %12, ptr noundef nonnull align 1 dereferenceable(7) %229, i64 7, i1 false)
  %230 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %231 = load ptr, ptr %230, align 8
  %232 = icmp eq i64 %224, 0
  br i1 %232, label %266, label %233

233:                                              ; preds = %227
  %234 = zext i8 %4 to i32
  %235 = icmp eq ptr %231, null
  %236 = getelementptr inbounds nuw i8, ptr %231, i64 64
  %237 = getelementptr inbounds nuw i8, ptr %231, i64 88
  br label %238

238:                                              ; preds = %262, %233
  %239 = phi i8 [ %228, %233 ], [ %263, %262 ]
  %240 = phi i64 [ %224, %233 ], [ %264, %262 ]
  br i1 %235, label %261, label %241

241:                                              ; preds = %238
  %242 = load ptr, ptr %236, align 8, !noalias !72
  %243 = load ptr, ptr %242, align 8, !noalias !72
  %244 = icmp eq ptr %243, null
  br i1 %244, label %254, label %245

245:                                              ; preds = %241
  %246 = load ptr, ptr %237, align 8, !noalias !72
  %247 = load i32, ptr %246, align 4, !noalias !72
  %248 = icmp sgt i32 %247, 0
  br i1 %248, label %249, label %254

249:                                              ; preds = %245
  %250 = add nsw i32 %247, -1
  store i32 %250, ptr %246, align 4, !noalias !72
  %251 = load ptr, ptr %236, align 8, !noalias !72
  %252 = load ptr, ptr %251, align 8, !noalias !72
  %253 = getelementptr inbounds nuw i8, ptr %252, i64 1
  store ptr %253, ptr %251, align 8, !noalias !72
  store i8 %4, ptr %252, align 1, !noalias !72
  br label %262

254:                                              ; preds = %245, %241
  %255 = load ptr, ptr %231, align 8, !noalias !72
  %256 = getelementptr inbounds nuw i8, ptr %255, i64 24
  %257 = load ptr, ptr %256, align 8, !noalias !72
  %258 = invoke noundef i32 %257(ptr noundef nonnull align 8 dereferenceable(104) %231, i32 noundef %234)
          to label %259 unwind label %583

259:                                              ; preds = %254
  %260 = icmp eq i32 %258, -1
  br i1 %260, label %261, label %262

261:                                              ; preds = %259, %238
  br label %262

262:                                              ; preds = %261, %259, %249
  %263 = phi i8 [ 1, %261 ], [ %239, %259 ], [ %239, %249 ]
  %264 = add i64 %240, -1
  %265 = icmp eq i64 %264, 0
  br i1 %265, label %266, label %238, !llvm.loop !54

266:                                              ; preds = %262, %227
  %267 = phi i8 [ %228, %227 ], [ %263, %262 ]
  store i8 %267, ptr %2, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %229, ptr noundef nonnull align 1 dereferenceable(7) %12, i64 7, i1 false)
  store ptr %231, ptr %230, align 8
  %268 = icmp eq i64 %42, 0
  br i1 %268, label %309, label %269

269:                                              ; preds = %266
  %270 = load ptr, ptr %9, align 8
  %271 = load i64, ptr %98, align 8
  %272 = icmp ugt i64 %271, 15
  %273 = select i1 %272, ptr %270, ptr %9
  %274 = icmp eq ptr %231, null
  %275 = getelementptr inbounds nuw i8, ptr %231, i64 64
  %276 = getelementptr inbounds nuw i8, ptr %231, i64 88
  br label %277

277:                                              ; preds = %269, %304
  %278 = phi i8 [ %305, %304 ], [ %267, %269 ]
  %279 = phi i64 [ %306, %304 ], [ %42, %269 ]
  %280 = phi ptr [ %307, %304 ], [ %273, %269 ]
  %281 = load i8, ptr %280, align 1, !noalias !75
  br i1 %274, label %303, label %282

282:                                              ; preds = %277
  %283 = load ptr, ptr %275, align 8, !noalias !75
  %284 = load ptr, ptr %283, align 8, !noalias !75
  %285 = icmp eq ptr %284, null
  br i1 %285, label %295, label %286

286:                                              ; preds = %282
  %287 = load ptr, ptr %276, align 8, !noalias !75
  %288 = load i32, ptr %287, align 4, !noalias !75
  %289 = icmp sgt i32 %288, 0
  br i1 %289, label %290, label %295

290:                                              ; preds = %286
  %291 = add nsw i32 %288, -1
  store i32 %291, ptr %287, align 4, !noalias !75
  %292 = load ptr, ptr %275, align 8, !noalias !75
  %293 = load ptr, ptr %292, align 8, !noalias !75
  %294 = getelementptr inbounds nuw i8, ptr %293, i64 1
  store ptr %294, ptr %292, align 8, !noalias !75
  store i8 %281, ptr %293, align 1, !noalias !75
  br label %304

295:                                              ; preds = %286, %282
  %296 = zext i8 %281 to i32
  %297 = load ptr, ptr %231, align 8, !noalias !75
  %298 = getelementptr inbounds nuw i8, ptr %297, i64 24
  %299 = load ptr, ptr %298, align 8, !noalias !75
  %300 = invoke noundef i32 %299(ptr noundef nonnull align 8 dereferenceable(104) %231, i32 noundef %296)
          to label %301 unwind label %583

301:                                              ; preds = %295
  %302 = icmp eq i32 %300, -1
  br i1 %302, label %303, label %304

303:                                              ; preds = %301, %277
  br label %304

304:                                              ; preds = %303, %301, %290
  %305 = phi i8 [ 1, %303 ], [ %278, %301 ], [ %278, %290 ]
  %306 = add i64 %279, -1
  %307 = getelementptr inbounds nuw i8, ptr %280, i64 1
  %308 = icmp eq i64 %306, 0
  br i1 %308, label %309, label %277, !llvm.loop !58

309:                                              ; preds = %304, %266
  %310 = phi i8 [ %267, %266 ], [ %305, %304 ]
  store i8 %310, ptr %2, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %229, ptr noundef nonnull align 1 dereferenceable(7) %12, i64 7, i1 false)
  store ptr %231, ptr %230, align 8
  br label %443

311:                                              ; preds = %215
  %312 = load i64, ptr %98, align 8
  %313 = load ptr, ptr %9, align 8
  %314 = load i8, ptr %2, align 8
  %315 = getelementptr inbounds nuw i8, ptr %2, i64 1
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %13, ptr noundef nonnull align 1 dereferenceable(7) %315, i64 7, i1 false)
  %316 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %317 = load ptr, ptr %316, align 8
  %318 = icmp eq i64 %42, 0
  br i1 %318, label %357, label %319

319:                                              ; preds = %311
  %320 = icmp ugt i64 %312, 15
  %321 = select i1 %320, ptr %313, ptr %9
  %322 = icmp eq ptr %317, null
  %323 = getelementptr inbounds nuw i8, ptr %317, i64 64
  %324 = getelementptr inbounds nuw i8, ptr %317, i64 88
  br label %325

325:                                              ; preds = %319, %352
  %326 = phi i8 [ %353, %352 ], [ %314, %319 ]
  %327 = phi i64 [ %354, %352 ], [ %42, %319 ]
  %328 = phi ptr [ %355, %352 ], [ %321, %319 ]
  %329 = load i8, ptr %328, align 1, !noalias !78
  br i1 %322, label %351, label %330

330:                                              ; preds = %325
  %331 = load ptr, ptr %323, align 8, !noalias !78
  %332 = load ptr, ptr %331, align 8, !noalias !78
  %333 = icmp eq ptr %332, null
  br i1 %333, label %343, label %334

334:                                              ; preds = %330
  %335 = load ptr, ptr %324, align 8, !noalias !78
  %336 = load i32, ptr %335, align 4, !noalias !78
  %337 = icmp sgt i32 %336, 0
  br i1 %337, label %338, label %343

338:                                              ; preds = %334
  %339 = add nsw i32 %336, -1
  store i32 %339, ptr %335, align 4, !noalias !78
  %340 = load ptr, ptr %323, align 8, !noalias !78
  %341 = load ptr, ptr %340, align 8, !noalias !78
  %342 = getelementptr inbounds nuw i8, ptr %341, i64 1
  store ptr %342, ptr %340, align 8, !noalias !78
  store i8 %329, ptr %341, align 1, !noalias !78
  br label %352

343:                                              ; preds = %334, %330
  %344 = zext i8 %329 to i32
  %345 = load ptr, ptr %317, align 8, !noalias !78
  %346 = getelementptr inbounds nuw i8, ptr %345, i64 24
  %347 = load ptr, ptr %346, align 8, !noalias !78
  %348 = invoke noundef i32 %347(ptr noundef nonnull align 8 dereferenceable(104) %317, i32 noundef %344)
          to label %349 unwind label %583

349:                                              ; preds = %343
  %350 = icmp eq i32 %348, -1
  br i1 %350, label %351, label %352

351:                                              ; preds = %349, %325
  br label %352

352:                                              ; preds = %351, %349, %338
  %353 = phi i8 [ 1, %351 ], [ %326, %349 ], [ %326, %338 ]
  %354 = add i64 %327, -1
  %355 = getelementptr inbounds nuw i8, ptr %328, i64 1
  %356 = icmp eq i64 %354, 0
  br i1 %356, label %357, label %325, !llvm.loop !58

357:                                              ; preds = %352, %311
  %358 = phi i8 [ %314, %311 ], [ %353, %352 ]
  store i8 %358, ptr %2, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %315, ptr noundef nonnull align 1 dereferenceable(7) %13, i64 7, i1 false)
  store ptr %317, ptr %316, align 8
  %359 = icmp eq i64 %224, 0
  br i1 %359, label %393, label %360

360:                                              ; preds = %357
  %361 = zext i8 %4 to i32
  %362 = icmp eq ptr %317, null
  %363 = getelementptr inbounds nuw i8, ptr %317, i64 64
  %364 = getelementptr inbounds nuw i8, ptr %317, i64 88
  br label %365

365:                                              ; preds = %389, %360
  %366 = phi i8 [ %358, %360 ], [ %390, %389 ]
  %367 = phi i64 [ %224, %360 ], [ %391, %389 ]
  br i1 %362, label %388, label %368

368:                                              ; preds = %365
  %369 = load ptr, ptr %363, align 8, !noalias !81
  %370 = load ptr, ptr %369, align 8, !noalias !81
  %371 = icmp eq ptr %370, null
  br i1 %371, label %381, label %372

372:                                              ; preds = %368
  %373 = load ptr, ptr %364, align 8, !noalias !81
  %374 = load i32, ptr %373, align 4, !noalias !81
  %375 = icmp sgt i32 %374, 0
  br i1 %375, label %376, label %381

376:                                              ; preds = %372
  %377 = add nsw i32 %374, -1
  store i32 %377, ptr %373, align 4, !noalias !81
  %378 = load ptr, ptr %363, align 8, !noalias !81
  %379 = load ptr, ptr %378, align 8, !noalias !81
  %380 = getelementptr inbounds nuw i8, ptr %379, i64 1
  store ptr %380, ptr %378, align 8, !noalias !81
  store i8 %4, ptr %379, align 1, !noalias !81
  br label %389

381:                                              ; preds = %372, %368
  %382 = load ptr, ptr %317, align 8, !noalias !81
  %383 = getelementptr inbounds nuw i8, ptr %382, i64 24
  %384 = load ptr, ptr %383, align 8, !noalias !81
  %385 = invoke noundef i32 %384(ptr noundef nonnull align 8 dereferenceable(104) %317, i32 noundef %361)
          to label %386 unwind label %583

386:                                              ; preds = %381
  %387 = icmp eq i32 %385, -1
  br i1 %387, label %388, label %389

388:                                              ; preds = %386, %365
  br label %389

389:                                              ; preds = %388, %386, %376
  %390 = phi i8 [ 1, %388 ], [ %366, %386 ], [ %366, %376 ]
  %391 = add i64 %367, -1
  %392 = icmp eq i64 %391, 0
  br i1 %392, label %393, label %365, !llvm.loop !54

393:                                              ; preds = %389, %357
  %394 = phi i8 [ %358, %357 ], [ %390, %389 ]
  store i8 %394, ptr %2, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %315, ptr noundef nonnull align 1 dereferenceable(7) %13, i64 7, i1 false)
  store ptr %317, ptr %316, align 8
  br label %443

395:                                              ; preds = %215
  %396 = load i64, ptr %98, align 8
  %397 = load ptr, ptr %9, align 8
  %398 = load i8, ptr %2, align 8
  %399 = getelementptr inbounds nuw i8, ptr %2, i64 1
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %14, ptr noundef nonnull align 1 dereferenceable(7) %399, i64 7, i1 false)
  %400 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %401 = load ptr, ptr %400, align 8
  %402 = icmp eq i64 %42, 0
  br i1 %402, label %441, label %403

403:                                              ; preds = %395
  %404 = icmp ugt i64 %396, 15
  %405 = select i1 %404, ptr %397, ptr %9
  %406 = icmp eq ptr %401, null
  %407 = getelementptr inbounds nuw i8, ptr %401, i64 64
  %408 = getelementptr inbounds nuw i8, ptr %401, i64 88
  br label %409

409:                                              ; preds = %403, %436
  %410 = phi i8 [ %437, %436 ], [ %398, %403 ]
  %411 = phi i64 [ %438, %436 ], [ %42, %403 ]
  %412 = phi ptr [ %439, %436 ], [ %405, %403 ]
  %413 = load i8, ptr %412, align 1, !noalias !84
  br i1 %406, label %435, label %414

414:                                              ; preds = %409
  %415 = load ptr, ptr %407, align 8, !noalias !84
  %416 = load ptr, ptr %415, align 8, !noalias !84
  %417 = icmp eq ptr %416, null
  br i1 %417, label %427, label %418

418:                                              ; preds = %414
  %419 = load ptr, ptr %408, align 8, !noalias !84
  %420 = load i32, ptr %419, align 4, !noalias !84
  %421 = icmp sgt i32 %420, 0
  br i1 %421, label %422, label %427

422:                                              ; preds = %418
  %423 = add nsw i32 %420, -1
  store i32 %423, ptr %419, align 4, !noalias !84
  %424 = load ptr, ptr %407, align 8, !noalias !84
  %425 = load ptr, ptr %424, align 8, !noalias !84
  %426 = getelementptr inbounds nuw i8, ptr %425, i64 1
  store ptr %426, ptr %424, align 8, !noalias !84
  store i8 %413, ptr %425, align 1, !noalias !84
  br label %436

427:                                              ; preds = %418, %414
  %428 = zext i8 %413 to i32
  %429 = load ptr, ptr %401, align 8, !noalias !84
  %430 = getelementptr inbounds nuw i8, ptr %429, i64 24
  %431 = load ptr, ptr %430, align 8, !noalias !84
  %432 = invoke noundef i32 %431(ptr noundef nonnull align 8 dereferenceable(104) %401, i32 noundef %428)
          to label %433 unwind label %583

433:                                              ; preds = %427
  %434 = icmp eq i32 %432, -1
  br i1 %434, label %435, label %436

435:                                              ; preds = %433, %409
  br label %436

436:                                              ; preds = %435, %433, %422
  %437 = phi i8 [ 1, %435 ], [ %410, %433 ], [ %410, %422 ]
  %438 = add i64 %411, -1
  %439 = getelementptr inbounds nuw i8, ptr %412, i64 1
  %440 = icmp eq i64 %438, 0
  br i1 %440, label %441, label %409, !llvm.loop !58

441:                                              ; preds = %436, %395
  %442 = phi i8 [ %398, %395 ], [ %437, %436 ]
  store i8 %442, ptr %2, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %399, ptr noundef nonnull align 1 dereferenceable(7) %14, i64 7, i1 false)
  store ptr %401, ptr %400, align 8
  br label %443

443:                                              ; preds = %393, %441, %309
  %444 = phi i64 [ 0, %309 ], [ 0, %393 ], [ %224, %441 ]
  %445 = sub i64 %217, %42
  %446 = load i64, ptr %98, align 8
  %447 = load ptr, ptr %9, align 8
  %448 = load i8, ptr %2, align 8
  %449 = getelementptr inbounds nuw i8, ptr %2, i64 1
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %15, ptr noundef nonnull align 1 dereferenceable(7) %449, i64 7, i1 false)
  %450 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %451 = load ptr, ptr %450, align 8
  %452 = icmp eq i64 %445, 0
  br i1 %452, label %492, label %453

453:                                              ; preds = %443
  %454 = icmp ugt i64 %446, 15
  %455 = select i1 %454, ptr %447, ptr %9
  %456 = getelementptr inbounds nuw i8, ptr %455, i64 %42
  %457 = icmp eq ptr %451, null
  %458 = getelementptr inbounds nuw i8, ptr %451, i64 64
  %459 = getelementptr inbounds nuw i8, ptr %451, i64 88
  br label %460

460:                                              ; preds = %453, %487
  %461 = phi i8 [ %488, %487 ], [ %448, %453 ]
  %462 = phi i64 [ %489, %487 ], [ %445, %453 ]
  %463 = phi ptr [ %490, %487 ], [ %456, %453 ]
  %464 = load i8, ptr %463, align 1, !noalias !87
  br i1 %457, label %486, label %465

465:                                              ; preds = %460
  %466 = load ptr, ptr %458, align 8, !noalias !87
  %467 = load ptr, ptr %466, align 8, !noalias !87
  %468 = icmp eq ptr %467, null
  br i1 %468, label %478, label %469

469:                                              ; preds = %465
  %470 = load ptr, ptr %459, align 8, !noalias !87
  %471 = load i32, ptr %470, align 4, !noalias !87
  %472 = icmp sgt i32 %471, 0
  br i1 %472, label %473, label %478

473:                                              ; preds = %469
  %474 = add nsw i32 %471, -1
  store i32 %474, ptr %470, align 4, !noalias !87
  %475 = load ptr, ptr %458, align 8, !noalias !87
  %476 = load ptr, ptr %475, align 8, !noalias !87
  %477 = getelementptr inbounds nuw i8, ptr %476, i64 1
  store ptr %477, ptr %475, align 8, !noalias !87
  store i8 %464, ptr %476, align 1, !noalias !87
  br label %487

478:                                              ; preds = %469, %465
  %479 = zext i8 %464 to i32
  %480 = load ptr, ptr %451, align 8, !noalias !87
  %481 = getelementptr inbounds nuw i8, ptr %480, i64 24
  %482 = load ptr, ptr %481, align 8, !noalias !87
  %483 = invoke noundef i32 %482(ptr noundef nonnull align 8 dereferenceable(104) %451, i32 noundef %479)
          to label %484 unwind label %583

484:                                              ; preds = %478
  %485 = icmp eq i32 %483, -1
  br i1 %485, label %486, label %487

486:                                              ; preds = %484, %460
  br label %487

487:                                              ; preds = %486, %484, %473
  %488 = phi i8 [ 1, %486 ], [ %461, %484 ], [ %461, %473 ]
  %489 = add i64 %462, -1
  %490 = getelementptr inbounds nuw i8, ptr %463, i64 1
  %491 = icmp eq i64 %489, 0
  br i1 %491, label %492, label %460, !llvm.loop !58

492:                                              ; preds = %487, %443
  %493 = phi i8 [ %448, %443 ], [ %488, %487 ]
  store i8 %493, ptr %2, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %449, ptr noundef nonnull align 1 dereferenceable(7) %15, i64 7, i1 false)
  store ptr %451, ptr %450, align 8
  store i64 0, ptr %218, align 8
  %494 = load i8, ptr %2, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %16, ptr noundef nonnull align 1 dereferenceable(7) %449, i64 7, i1 false)
  %495 = load ptr, ptr %450, align 8
  %496 = icmp eq i64 %444, 0
  br i1 %496, label %530, label %497

497:                                              ; preds = %492
  %498 = zext i8 %4 to i32
  %499 = icmp eq ptr %495, null
  %500 = getelementptr inbounds nuw i8, ptr %495, i64 64
  %501 = getelementptr inbounds nuw i8, ptr %495, i64 88
  br label %502

502:                                              ; preds = %526, %497
  %503 = phi i8 [ %494, %497 ], [ %527, %526 ]
  %504 = phi i64 [ %444, %497 ], [ %528, %526 ]
  br i1 %499, label %525, label %505

505:                                              ; preds = %502
  %506 = load ptr, ptr %500, align 8, !noalias !90
  %507 = load ptr, ptr %506, align 8, !noalias !90
  %508 = icmp eq ptr %507, null
  br i1 %508, label %518, label %509

509:                                              ; preds = %505
  %510 = load ptr, ptr %501, align 8, !noalias !90
  %511 = load i32, ptr %510, align 4, !noalias !90
  %512 = icmp sgt i32 %511, 0
  br i1 %512, label %513, label %518

513:                                              ; preds = %509
  %514 = add nsw i32 %511, -1
  store i32 %514, ptr %510, align 4, !noalias !90
  %515 = load ptr, ptr %500, align 8, !noalias !90
  %516 = load ptr, ptr %515, align 8, !noalias !90
  %517 = getelementptr inbounds nuw i8, ptr %516, i64 1
  store ptr %517, ptr %515, align 8, !noalias !90
  store i8 %4, ptr %516, align 1, !noalias !90
  br label %526

518:                                              ; preds = %509, %505
  %519 = load ptr, ptr %495, align 8, !noalias !90
  %520 = getelementptr inbounds nuw i8, ptr %519, i64 24
  %521 = load ptr, ptr %520, align 8, !noalias !90
  %522 = invoke noundef i32 %521(ptr noundef nonnull align 8 dereferenceable(104) %495, i32 noundef %498)
          to label %523 unwind label %583

523:                                              ; preds = %518
  %524 = icmp eq i32 %522, -1
  br i1 %524, label %525, label %526

525:                                              ; preds = %523, %502
  br label %526

526:                                              ; preds = %525, %523, %513
  %527 = phi i8 [ 1, %525 ], [ %503, %523 ], [ %503, %513 ]
  %528 = add i64 %504, -1
  %529 = icmp eq i64 %528, 0
  br i1 %529, label %530, label %502, !llvm.loop !54

530:                                              ; preds = %526, %492
  %531 = phi i8 [ %494, %492 ], [ %527, %526 ]
  store i8 %531, ptr %1, align 8
  %532 = getelementptr inbounds nuw i8, ptr %1, i64 1
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %532, ptr noundef nonnull align 1 dereferenceable(7) %16, i64 7, i1 false)
  %533 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store ptr %495, ptr %533, align 8
  %534 = load i64, ptr %135, align 8
  %535 = icmp ugt i64 %534, 15
  br i1 %535, label %536, label %557

536:                                              ; preds = %530
  %537 = load ptr, ptr %11, align 8
  %538 = add i64 %534, 1
  %539 = icmp ugt i64 %538, 4095
  br i1 %539, label %540, label %554

540:                                              ; preds = %536
  %541 = getelementptr inbounds i8, ptr %537, i64 -8
  %542 = load i64, ptr %541, align 8
  %543 = ptrtoint ptr %537 to i64
  %544 = add i64 %543, -8
  %545 = sub i64 %544, %542
  %546 = icmp ult i64 %545, 32
  br i1 %546, label %549, label %547

547:                                              ; preds = %540
  invoke void @_invalid_parameter_noinfo_noreturn() #29
          to label %548 unwind label %552

548:                                              ; preds = %547
  unreachable

549:                                              ; preds = %540
  %550 = add i64 %534, 40
  %551 = inttoptr i64 %542 to ptr
  br label %554

552:                                              ; preds = %547
  %553 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %553) ]
  unreachable

554:                                              ; preds = %549, %536
  %555 = phi i64 [ %550, %549 ], [ %538, %536 ]
  %556 = phi ptr [ %551, %549 ], [ %537, %536 ]
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %556, i64 noundef %555) #10
  br label %557

557:                                              ; preds = %530, %554
  %558 = getelementptr inbounds nuw i8, ptr %11, i64 16
  store i64 0, ptr %558, align 8
  store i64 15, ptr %135, align 8
  store i8 0, ptr %11, align 8
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %11) #10
  %559 = load i64, ptr %98, align 8
  %560 = icmp ugt i64 %559, 15
  br i1 %560, label %561, label %582

561:                                              ; preds = %557
  %562 = load ptr, ptr %9, align 8
  %563 = add i64 %559, 1
  %564 = icmp ugt i64 %563, 4095
  br i1 %564, label %565, label %579

565:                                              ; preds = %561
  %566 = getelementptr inbounds i8, ptr %562, i64 -8
  %567 = load i64, ptr %566, align 8
  %568 = ptrtoint ptr %562 to i64
  %569 = add i64 %568, -8
  %570 = sub i64 %569, %567
  %571 = icmp ult i64 %570, 32
  br i1 %571, label %574, label %572

572:                                              ; preds = %565
  invoke void @_invalid_parameter_noinfo_noreturn() #29
          to label %573 unwind label %577

573:                                              ; preds = %572
  unreachable

574:                                              ; preds = %565
  %575 = add i64 %559, 40
  %576 = inttoptr i64 %567 to ptr
  br label %579

577:                                              ; preds = %572
  %578 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %578) ]
  unreachable

579:                                              ; preds = %574, %561
  %580 = phi i64 [ %575, %574 ], [ %563, %561 ]
  %581 = phi ptr [ %576, %574 ], [ %562, %561 ]
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %581, i64 noundef %580) #10
  br label %582

582:                                              ; preds = %557, %579
  store i64 0, ptr %216, align 8
  store i64 15, ptr %98, align 8
  store i8 0, ptr %9, align 8
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %9) #10
  ret void

583:                                              ; preds = %518, %478, %427, %381, %343, %295, %254, %178, %164, %143
  %584 = cleanuppad within none []
  call void @"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %11) #10 [ "funclet"(token %584) ]
  cleanupret from %584 unwind label %585

585:                                              ; preds = %130, %97, %214, %583
  %586 = cleanuppad within none []
  call void @"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %9) #10 [ "funclet"(token %586) ]
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %9) #10
  cleanupret from %586 unwind to caller
}

; Function Attrs: inlinehint mustprogress uwtable
define linkonce_odr dso_local i32 @sprintf_s(ptr noundef %0, i64 noundef %1, ptr noundef %2, ...) local_unnamed_addr #15 comdat {
  %4 = alloca ptr, align 8
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %4) #10
  call void @llvm.va_start.p0(ptr nonnull %4)
  %5 = load ptr, ptr %4, align 8
  %6 = call ptr @__local_stdio_printf_options()
  %7 = load i64, ptr %6, align 8
  %8 = call i32 @__stdio_common_vsprintf_s(i64 noundef %7, ptr noundef %0, i64 noundef %1, ptr noundef %2, ptr noundef null, ptr noundef %5)
  %9 = call i32 @llvm.smax.i32(i32 %8, i32 -1)
  call void @llvm.va_end.p0(ptr %4)
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %4) #10
  ret i32 %9
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(48) ptr @"??$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0) local_unnamed_addr #2 comdat personality ptr @__CxxFrameHandler3 {
  %2 = alloca %"class.std::_Lockit", align 4
  %3 = alloca %"class.std::_Lockit", align 4
  %4 = alloca ptr, align 8
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %3) #10
  %5 = call noundef ptr @"??0_Lockit@std@@QEAA@H@Z"(ptr noundef nonnull align 4 dereferenceable(4) %3, i32 noundef 0) #10
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %4) #10
  %6 = load ptr, ptr @"?_Psave@?$_Facetptr@V?$numpunct@D@std@@@std@@2PEBVfacet@locale@2@EB", align 8
  store ptr %6, ptr %4, align 8
  %7 = load i64, ptr @"?id@?$numpunct@D@std@@2V0locale@2@A", align 8
  %8 = icmp eq i64 %7, 0
  br i1 %8, label %9, label %18

9:                                                ; preds = %1
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %2) #10
  %10 = call noundef ptr @"??0_Lockit@std@@QEAA@H@Z"(ptr noundef nonnull align 4 dereferenceable(4) %2, i32 noundef 0) #10
  %11 = load i64, ptr @"?id@?$numpunct@D@std@@2V0locale@2@A", align 8
  %12 = icmp eq i64 %11, 0
  br i1 %12, label %13, label %17

13:                                               ; preds = %9
  %14 = load i32, ptr @"?_Id_cnt@id@locale@std@@0HA", align 4
  %15 = add nsw i32 %14, 1
  store i32 %15, ptr @"?_Id_cnt@id@locale@std@@0HA", align 4
  %16 = sext i32 %15 to i64
  store i64 %16, ptr @"?id@?$numpunct@D@std@@2V0locale@2@A", align 8
  br label %17

17:                                               ; preds = %13, %9
  call void @"??1_Lockit@std@@QEAA@XZ"(ptr noundef nonnull align 4 dereferenceable(4) %2) #10
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %2) #10
  br label %18

18:                                               ; preds = %1, %17
  %19 = load i64, ptr @"?id@?$numpunct@D@std@@2V0locale@2@A", align 8
  %20 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %21 = load ptr, ptr %20, align 8
  %22 = getelementptr inbounds nuw i8, ptr %21, i64 24
  %23 = load i64, ptr %22, align 8
  %24 = icmp ult i64 %19, %23
  br i1 %24, label %25, label %30

25:                                               ; preds = %18
  %26 = getelementptr inbounds nuw i8, ptr %21, i64 16
  %27 = load ptr, ptr %26, align 8
  %28 = getelementptr inbounds nuw ptr, ptr %27, i64 %19
  %29 = load ptr, ptr %28, align 8
  br label %30

30:                                               ; preds = %25, %18
  %31 = phi ptr [ %29, %25 ], [ null, %18 ]
  %32 = icmp eq ptr %31, null
  br i1 %32, label %33, label %48

33:                                               ; preds = %30
  %34 = getelementptr inbounds nuw i8, ptr %21, i64 36
  %35 = load i8, ptr %34, align 4, !range !18, !noundef !19
  %36 = trunc nuw i8 %35 to i1
  br i1 %36, label %37, label %48

37:                                               ; preds = %33
  %38 = invoke noundef ptr @"?_Getgloballocale@locale@std@@CAPEAV_Locimp@12@XZ"()
          to label %39 unwind label %77

39:                                               ; preds = %37
  %40 = getelementptr inbounds nuw i8, ptr %38, i64 24
  %41 = load i64, ptr %40, align 8
  %42 = icmp ult i64 %19, %41
  br i1 %42, label %43, label %48

43:                                               ; preds = %39
  %44 = getelementptr inbounds nuw i8, ptr %38, i64 16
  %45 = load ptr, ptr %44, align 8
  %46 = getelementptr inbounds nuw ptr, ptr %45, i64 %19
  %47 = load ptr, ptr %46, align 8
  br label %48

48:                                               ; preds = %43, %39, %33, %30
  %49 = phi ptr [ %31, %33 ], [ %31, %30 ], [ %47, %43 ], [ null, %39 ]
  %50 = icmp eq ptr %49, null
  br i1 %50, label %51, label %75

51:                                               ; preds = %48
  %52 = load ptr, ptr %4, align 8
  %53 = icmp eq ptr %52, null
  br i1 %53, label %54, label %75

54:                                               ; preds = %51
  %55 = invoke noundef i64 @"?_Getcat@?$numpunct@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"(ptr noundef nonnull %4, ptr noundef nonnull %0)
          to label %56 unwind label %77

56:                                               ; preds = %54
  %57 = icmp eq i64 %55, -1
  br i1 %57, label %58, label %60

58:                                               ; preds = %56
  invoke void @"?_Throw_bad_cast@std@@YAXXZ"() #29
          to label %59 unwind label %77

59:                                               ; preds = %58
  unreachable

60:                                               ; preds = %56
  %61 = load ptr, ptr %4, align 8
  invoke void @"?_Facet_Register@std@@YAXPEAV_Facet_base@1@@Z"(ptr noundef %61)
          to label %62 unwind label %67

62:                                               ; preds = %60
  %63 = load ptr, ptr %61, align 8
  %64 = getelementptr inbounds nuw i8, ptr %63, i64 8
  %65 = load ptr, ptr %64, align 8
  call void %65(ptr noundef nonnull align 8 dereferenceable(16) %61) #10
  %66 = load ptr, ptr %4, align 8
  store ptr %66, ptr @"?_Psave@?$_Facetptr@V?$numpunct@D@std@@@std@@2PEBVfacet@locale@2@EB", align 8
  br label %75

67:                                               ; preds = %60
  %68 = cleanuppad within none []
  %69 = icmp eq ptr %61, null
  br i1 %69, label %74, label %70

70:                                               ; preds = %67
  %71 = load ptr, ptr %61, align 8
  %72 = load ptr, ptr %71, align 8
  %73 = call noundef ptr %72(ptr noundef nonnull align 8 dereferenceable(8) %61, i32 noundef 1) #10 [ "funclet"(token %68) ]
  br label %74

74:                                               ; preds = %67, %70
  cleanupret from %68 unwind label %77

75:                                               ; preds = %51, %62, %48
  %76 = phi ptr [ %49, %48 ], [ %66, %62 ], [ %52, %51 ]
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %4) #10
  call void @"??1_Lockit@std@@QEAA@XZ"(ptr noundef nonnull align 4 dereferenceable(4) %3) #10
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %3) #10
  ret ptr %76

77:                                               ; preds = %37, %74, %58, %54
  %78 = cleanuppad within none []
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %4) #10
  call void @"??1_Lockit@std@@QEAA@XZ"(ptr noundef nonnull align 4 dereferenceable(4) %3) #10 [ "funclet"(token %78) ]
  cleanupret from %78 unwind to caller
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef i64 @"?_Getcat@?$numpunct@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"(ptr noundef %0, ptr noundef %1) local_unnamed_addr #2 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %3 = alloca %"class.std::_Locinfo", align 8
  %4 = icmp eq ptr %0, null
  br i1 %4, label %29, label %5

5:                                                ; preds = %2
  %6 = load ptr, ptr %0, align 8
  %7 = icmp eq ptr %6, null
  br i1 %7, label %8, label %29

8:                                                ; preds = %5
  %9 = tail call noalias noundef nonnull dereferenceable(48) ptr @"??2@YAPEAX_K@Z"(i64 noundef 48) #27
  call void @llvm.lifetime.start.p0(i64 104, ptr nonnull %3) #10
  %10 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %11 = load ptr, ptr %10, align 8
  %12 = icmp eq ptr %11, null
  br i1 %12, label %19, label %13

13:                                               ; preds = %8
  %14 = getelementptr inbounds nuw i8, ptr %11, i64 40
  %15 = load ptr, ptr %14, align 8
  %16 = icmp eq ptr %15, null
  %17 = getelementptr inbounds nuw i8, ptr %11, i64 48
  %18 = select i1 %16, ptr %17, ptr %15
  br label %19

19:                                               ; preds = %8, %13
  %20 = phi ptr [ %18, %13 ], [ @"??_C@_00CNPNBAHC@?$AA@", %8 ]
  %21 = invoke noundef ptr @"??0_Locinfo@std@@QEAA@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(104) %3, ptr noundef %20)
          to label %22 unwind label %27

22:                                               ; preds = %19
  store ptr @"??_7facet@locale@std@@6B@", ptr %9, align 8
  %23 = getelementptr inbounds nuw i8, ptr %9, i64 8
  store i32 0, ptr %23, align 8
  store ptr @"??_7?$numpunct@D@std@@6B@", ptr %9, align 8
  invoke void @"?_Init@?$numpunct@D@std@@IEAAXAEBV_Locinfo@2@_N@Z"(ptr noundef nonnull align 8 dereferenceable(48) %9, ptr noundef nonnull align 8 dereferenceable(104) %3, i1 noundef zeroext true)
          to label %24 unwind label %25

24:                                               ; preds = %22
  store ptr %9, ptr %0, align 8
  call void @"??1_Locinfo@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(104) %3) #10
  call void @llvm.lifetime.end.p0(i64 104, ptr nonnull %3) #10
  br label %29

25:                                               ; preds = %22
  %26 = cleanuppad within none []
  call void @"??1_Locinfo@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(104) %3) #10 [ "funclet"(token %26) ]
  cleanupret from %26 unwind label %27

27:                                               ; preds = %25, %19
  %28 = cleanuppad within none []
  call void @llvm.lifetime.end.p0(i64 104, ptr nonnull %3) #10
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %9, i64 noundef 48) #28 [ "funclet"(token %28) ]
  cleanupret from %28 unwind to caller

29:                                               ; preds = %24, %5, %2
  ret i64 4
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @"?_Init@?$numpunct@D@std@@IEAAXAEBV_Locinfo@2@_N@Z"(ptr noundef nonnull align 8 dereferenceable(48) %0, ptr noundef nonnull align 8 dereferenceable(104) %1, i1 noundef zeroext %2) local_unnamed_addr #2 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %4 = alloca %struct._Cvtvec, align 4
  %5 = alloca %"struct.std::_Tidy_guard", align 8
  %6 = tail call noundef ptr @localeconv()
  call void @llvm.lifetime.start.p0(i64 44, ptr nonnull %4) #10
  call void @_Getcvt(ptr dead_on_unwind nonnull writable sret(%struct._Cvtvec) align 4 %4) #10
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store ptr null, ptr %7, align 8
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 40
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %8, i8 0, i64 16, i1 false)
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %5) #10
  store ptr %0, ptr %5, align 8
  br i1 %2, label %13, label %10

10:                                               ; preds = %3
  %11 = getelementptr inbounds nuw i8, ptr %6, i64 16
  %12 = load ptr, ptr %11, align 8
  br label %13

13:                                               ; preds = %3, %10
  %14 = phi ptr [ %12, %10 ], [ @"??_C@_00CNPNBAHC@?$AA@", %3 ]
  %15 = call i64 @strlen(ptr noundef nonnull dereferenceable(1) %14) #10
  %16 = add i64 %15, 1
  %17 = call noalias ptr @calloc(i64 noundef %16, i64 noundef 1) #32
  %18 = icmp eq ptr %17, null
  br i1 %18, label %22, label %19

19:                                               ; preds = %13
  %20 = icmp eq i64 %16, 0
  br i1 %20, label %24, label %21

21:                                               ; preds = %19
  call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 1 %17, ptr nonnull align 1 %14, i64 %16, i1 false)
  br label %24

22:                                               ; preds = %13
  invoke void @"?_Xbad_alloc@std@@YAXXZ"() #29
          to label %23 unwind label %47

23:                                               ; preds = %22
  unreachable

24:                                               ; preds = %21, %19
  store ptr %17, ptr %7, align 8
  %25 = call noalias dereferenceable_or_null(6) ptr @calloc(i64 noundef 6, i64 noundef 1) #32
  %26 = icmp eq ptr %25, null
  br i1 %26, label %30, label %27

27:                                               ; preds = %24
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(6) %25, ptr noundef nonnull align 1 dereferenceable(6) @"??_C@_05LAPONLG@false?$AA@", i64 6, i1 false)
  store ptr %25, ptr %8, align 8
  %28 = call noalias dereferenceable_or_null(5) ptr @calloc(i64 noundef 5, i64 noundef 1) #32
  %29 = icmp eq ptr %28, null
  br i1 %29, label %33, label %32

30:                                               ; preds = %24
  invoke void @"?_Xbad_alloc@std@@YAXXZ"() #29
          to label %31 unwind label %47

31:                                               ; preds = %30
  unreachable

32:                                               ; preds = %27
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(5) %28, ptr noundef nonnull align 1 dereferenceable(5) @"??_C@_04LOAJBDKD@true?$AA@", i64 5, i1 false)
  store ptr %28, ptr %9, align 8
  store ptr null, ptr %5, align 8
  br i1 %2, label %35, label %37

33:                                               ; preds = %27
  invoke void @"?_Xbad_alloc@std@@YAXXZ"() #29
          to label %34 unwind label %47

34:                                               ; preds = %33
  unreachable

35:                                               ; preds = %32
  %36 = getelementptr inbounds nuw i8, ptr %0, i64 24
  store i8 46, ptr %36, align 8
  br label %44

37:                                               ; preds = %32
  %38 = load ptr, ptr %6, align 8
  %39 = load i8, ptr %38, align 1
  %40 = getelementptr inbounds nuw i8, ptr %0, i64 24
  store i8 %39, ptr %40, align 8
  %41 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %42 = load ptr, ptr %41, align 8
  %43 = load i8, ptr %42, align 1
  br label %44

44:                                               ; preds = %35, %37
  %45 = phi i8 [ 44, %35 ], [ %43, %37 ]
  %46 = getelementptr inbounds nuw i8, ptr %0, i64 25
  store i8 %45, ptr %46, align 1
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %5) #10
  call void @llvm.lifetime.end.p0(i64 44, ptr nonnull %4) #10
  ret void

47:                                               ; preds = %33, %30, %22
  %48 = cleanuppad within none []
  call void @"??1?$_Tidy_guard@V?$numpunct@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %5) #10 [ "funclet"(token %48) ]
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %5) #10
  call void @llvm.lifetime.end.p0(i64 44, ptr nonnull %4) #10
  cleanupret from %48 unwind to caller
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local noundef ptr @"??_G?$numpunct@D@std@@MEAAPEAXI@Z"(ptr noundef nonnull align 8 dereferenceable(48) %0, i32 noundef %1) unnamed_addr #7 comdat align 2 personality ptr @__CxxFrameHandler3 {
  store ptr @"??_7?$numpunct@D@std@@6B@", ptr %0, align 8
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %4 = load ptr, ptr %3, align 8
  tail call void @free(ptr noundef %4)
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %6 = load ptr, ptr %5, align 8
  tail call void @free(ptr noundef %6)
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 40
  %8 = load ptr, ptr %7, align 8
  tail call void @free(ptr noundef %8)
  %9 = icmp eq i32 %1, 0
  br i1 %9, label %11, label %10

10:                                               ; preds = %2
  tail call void @"??3@YAXPEAX_K@Z"(ptr noundef %0, i64 noundef 48) #28
  br label %11

11:                                               ; preds = %10, %2
  ret ptr %0
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local noundef i8 @"?do_decimal_point@?$numpunct@D@std@@MEBADXZ"(ptr noundef nonnull align 8 dereferenceable(48) %0) unnamed_addr #7 comdat align 2 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %3 = load i8, ptr %2, align 8
  ret i8 %3
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local noundef i8 @"?do_thousands_sep@?$numpunct@D@std@@MEBADXZ"(ptr noundef nonnull align 8 dereferenceable(48) %0) unnamed_addr #7 comdat align 2 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 25
  %3 = load i8, ptr %2, align 1
  ret i8 %3
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @"?do_grouping@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"(ptr noundef nonnull align 8 dereferenceable(48) %0, ptr dead_on_unwind noalias writable sret(%"class.std::basic_string") align 8 %1) unnamed_addr #2 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %4 = load ptr, ptr %3, align 8
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %1, i8 0, i64 32, i1 false)
  %5 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %4) #10
  %6 = icmp slt i64 %5, 0
  br i1 %6, label %7, label %8

7:                                                ; preds = %2
  tail call void @"?_Xlen_string@std@@YAXXZ"() #29
  unreachable

8:                                                ; preds = %2
  %9 = icmp ult i64 %5, 16
  br i1 %9, label %10, label %14

10:                                               ; preds = %8
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store i64 %5, ptr %11, align 8
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 24
  store i64 15, ptr %12, align 8
  tail call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 8 dereferenceable(32) %1, ptr nonnull align 1 %4, i64 %5, i1 false)
  %13 = getelementptr inbounds nuw [16 x i8], ptr %1, i64 0, i64 %5
  store i8 0, ptr %13, align 1
  br label %37

14:                                               ; preds = %8
  %15 = or i64 %5, 15
  %16 = tail call i64 @llvm.umax.i64(i64 %15, i64 22)
  %17 = icmp ugt i64 %15, 4094
  br i1 %17, label %18, label %29

18:                                               ; preds = %14
  %19 = icmp ult i64 %15, -40
  br i1 %19, label %21, label %20

20:                                               ; preds = %18
  tail call void @"?_Throw_bad_array_new_length@std@@YAXXZ"() #29
  unreachable

21:                                               ; preds = %18
  %22 = add nuw i64 %16, 40
  %23 = tail call noalias noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef %22) #31
  %24 = ptrtoint ptr %23 to i64
  %25 = add i64 %24, 39
  %26 = and i64 %25, -32
  %27 = inttoptr i64 %26 to ptr
  %28 = getelementptr inbounds i8, ptr %27, i64 -8
  store i64 %24, ptr %28, align 8
  br label %32

29:                                               ; preds = %14
  %30 = add nuw nsw i64 %16, 1
  %31 = tail call noalias noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef %30) #31
  br label %32

32:                                               ; preds = %29, %21
  %33 = phi ptr [ %27, %21 ], [ %31, %29 ]
  store ptr %33, ptr %1, align 8
  %34 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store i64 %5, ptr %34, align 8
  %35 = getelementptr inbounds nuw i8, ptr %1, i64 24
  store i64 %16, ptr %35, align 8
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 1 %33, ptr nonnull align 1 %4, i64 %5, i1 false)
  %36 = getelementptr inbounds nuw i8, ptr %33, i64 %5
  store i8 0, ptr %36, align 1
  br label %37

37:                                               ; preds = %10, %32
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @"?do_falsename@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"(ptr noundef nonnull align 8 dereferenceable(48) %0, ptr dead_on_unwind noalias writable sret(%"class.std::basic_string") align 8 %1) unnamed_addr #2 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %4 = load ptr, ptr %3, align 8
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %1, i8 0, i64 32, i1 false)
  %5 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %4) #10
  %6 = icmp slt i64 %5, 0
  br i1 %6, label %7, label %8

7:                                                ; preds = %2
  tail call void @"?_Xlen_string@std@@YAXXZ"() #29
  unreachable

8:                                                ; preds = %2
  %9 = icmp ult i64 %5, 16
  br i1 %9, label %10, label %14

10:                                               ; preds = %8
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store i64 %5, ptr %11, align 8
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 24
  store i64 15, ptr %12, align 8
  tail call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 8 dereferenceable(32) %1, ptr nonnull align 1 %4, i64 %5, i1 false)
  %13 = getelementptr inbounds nuw [16 x i8], ptr %1, i64 0, i64 %5
  store i8 0, ptr %13, align 1
  br label %37

14:                                               ; preds = %8
  %15 = or i64 %5, 15
  %16 = tail call i64 @llvm.umax.i64(i64 %15, i64 22)
  %17 = icmp ugt i64 %15, 4094
  br i1 %17, label %18, label %29

18:                                               ; preds = %14
  %19 = icmp ult i64 %15, -40
  br i1 %19, label %21, label %20

20:                                               ; preds = %18
  tail call void @"?_Throw_bad_array_new_length@std@@YAXXZ"() #29
  unreachable

21:                                               ; preds = %18
  %22 = add nuw i64 %16, 40
  %23 = tail call noalias noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef %22) #31
  %24 = ptrtoint ptr %23 to i64
  %25 = add i64 %24, 39
  %26 = and i64 %25, -32
  %27 = inttoptr i64 %26 to ptr
  %28 = getelementptr inbounds i8, ptr %27, i64 -8
  store i64 %24, ptr %28, align 8
  br label %32

29:                                               ; preds = %14
  %30 = add nuw nsw i64 %16, 1
  %31 = tail call noalias noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef %30) #31
  br label %32

32:                                               ; preds = %29, %21
  %33 = phi ptr [ %27, %21 ], [ %31, %29 ]
  store ptr %33, ptr %1, align 8
  %34 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store i64 %5, ptr %34, align 8
  %35 = getelementptr inbounds nuw i8, ptr %1, i64 24
  store i64 %16, ptr %35, align 8
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 1 %33, ptr nonnull align 1 %4, i64 %5, i1 false)
  %36 = getelementptr inbounds nuw i8, ptr %33, i64 %5
  store i8 0, ptr %36, align 1
  br label %37

37:                                               ; preds = %10, %32
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @"?do_truename@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"(ptr noundef nonnull align 8 dereferenceable(48) %0, ptr dead_on_unwind noalias writable sret(%"class.std::basic_string") align 8 %1) unnamed_addr #2 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 40
  %4 = load ptr, ptr %3, align 8
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %1, i8 0, i64 32, i1 false)
  %5 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %4) #10
  %6 = icmp slt i64 %5, 0
  br i1 %6, label %7, label %8

7:                                                ; preds = %2
  tail call void @"?_Xlen_string@std@@YAXXZ"() #29
  unreachable

8:                                                ; preds = %2
  %9 = icmp ult i64 %5, 16
  br i1 %9, label %10, label %14

10:                                               ; preds = %8
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store i64 %5, ptr %11, align 8
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 24
  store i64 15, ptr %12, align 8
  tail call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 8 dereferenceable(32) %1, ptr nonnull align 1 %4, i64 %5, i1 false)
  %13 = getelementptr inbounds nuw [16 x i8], ptr %1, i64 0, i64 %5
  store i8 0, ptr %13, align 1
  br label %37

14:                                               ; preds = %8
  %15 = or i64 %5, 15
  %16 = tail call i64 @llvm.umax.i64(i64 %15, i64 22)
  %17 = icmp ugt i64 %15, 4094
  br i1 %17, label %18, label %29

18:                                               ; preds = %14
  %19 = icmp ult i64 %15, -40
  br i1 %19, label %21, label %20

20:                                               ; preds = %18
  tail call void @"?_Throw_bad_array_new_length@std@@YAXXZ"() #29
  unreachable

21:                                               ; preds = %18
  %22 = add nuw i64 %16, 40
  %23 = tail call noalias noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef %22) #31
  %24 = ptrtoint ptr %23 to i64
  %25 = add i64 %24, 39
  %26 = and i64 %25, -32
  %27 = inttoptr i64 %26 to ptr
  %28 = getelementptr inbounds i8, ptr %27, i64 -8
  store i64 %24, ptr %28, align 8
  br label %32

29:                                               ; preds = %14
  %30 = add nuw nsw i64 %16, 1
  %31 = tail call noalias noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef %30) #31
  br label %32

32:                                               ; preds = %29, %21
  %33 = phi ptr [ %27, %21 ], [ %31, %29 ]
  store ptr %33, ptr %1, align 8
  %34 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store i64 %5, ptr %34, align 8
  %35 = getelementptr inbounds nuw i8, ptr %1, i64 24
  store i64 %16, ptr %35, align 8
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 1 %33, ptr nonnull align 1 %4, i64 %5, i1 false)
  %36 = getelementptr inbounds nuw i8, ptr %33, i64 %5
  store i8 0, ptr %36, align 1
  br label %37

37:                                               ; preds = %10, %32
  ret void
}

; Function Attrs: inlinehint mustprogress nounwind uwtable
define linkonce_odr dso_local void @"??1?$_Tidy_guard@V?$numpunct@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) unnamed_addr #9 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp eq ptr %2, null
  br i1 %3, label %11, label %4

4:                                                ; preds = %1
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %6 = load ptr, ptr %5, align 8
  tail call void @free(ptr noundef %6)
  %7 = getelementptr inbounds nuw i8, ptr %2, i64 32
  %8 = load ptr, ptr %7, align 8
  tail call void @free(ptr noundef %8)
  %9 = getelementptr inbounds nuw i8, ptr %2, i64 40
  %10 = load ptr, ptr %9, align 8
  tail call void @free(ptr noundef %10)
  br label %11

11:                                               ; preds = %4, %1
  ret void
}

declare dso_local ptr @localeconv() local_unnamed_addr #11

; Function Attrs: nounwind
declare dso_local void @_Getcvt(ptr dead_on_unwind writable sret(%struct._Cvtvec) align 4) local_unnamed_addr #16

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite)
declare dso_local noalias noundef ptr @calloc(i64 noundef, i64 noundef) local_unnamed_addr #19

; Function Attrs: noreturn
declare dso_local void @"?_Xbad_alloc@std@@YAXXZ"() local_unnamed_addr #14

; Function Attrs: inlinehint mustprogress uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(32) ptr @"??$_Reallocate_grow_by@V<lambda_1>@?0??insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@_K0D@Z@_K_KD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??insert@01@QEAAAEAV01@00D@Z@_K2D@Z"(ptr noundef nonnull align 8 dereferenceable(32) %0, i64 noundef %1, i8 %2, i64 noundef %3, i64 noundef %4, i8 noundef %5) local_unnamed_addr #15 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %8 = load i64, ptr %7, align 8
  %9 = sub i64 9223372036854775807, %8
  %10 = icmp ult i64 %9, %1
  br i1 %10, label %11, label %12

11:                                               ; preds = %6
  tail call void @"?_Xlen_string@std@@YAXXZ"() #29
  unreachable

12:                                               ; preds = %6
  %13 = add i64 %8, %1
  %14 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %15 = load i64, ptr %14, align 8
  %16 = or i64 %13, 15
  %17 = icmp slt i64 %13, 0
  br i1 %17, label %25, label %18

18:                                               ; preds = %12
  %19 = lshr i64 %15, 1
  %20 = xor i64 %19, 9223372036854775807
  %21 = icmp ugt i64 %15, %20
  br i1 %21, label %25, label %22

22:                                               ; preds = %18
  %23 = add i64 %19, %15
  %24 = tail call i64 @llvm.umax.i64(i64 %16, i64 %23)
  br label %25

25:                                               ; preds = %12, %18, %22
  %26 = phi i64 [ %24, %22 ], [ 9223372036854775807, %12 ], [ 9223372036854775807, %18 ]
  %27 = add i64 %26, 1
  %28 = icmp eq i64 %27, 0
  br i1 %28, label %44, label %29

29:                                               ; preds = %25
  %30 = icmp ugt i64 %27, 4095
  br i1 %30, label %31, label %42

31:                                               ; preds = %29
  %32 = icmp ult i64 %27, -39
  br i1 %32, label %34, label %33

33:                                               ; preds = %31
  tail call void @"?_Throw_bad_array_new_length@std@@YAXXZ"() #29
  unreachable

34:                                               ; preds = %31
  %35 = add i64 %26, 40
  %36 = tail call noalias noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef %35) #31
  %37 = ptrtoint ptr %36 to i64
  %38 = add i64 %37, 39
  %39 = and i64 %38, -32
  %40 = inttoptr i64 %39 to ptr
  %41 = getelementptr inbounds i8, ptr %40, i64 -8
  store i64 %37, ptr %41, align 8
  br label %44

42:                                               ; preds = %29
  %43 = tail call noalias noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef %27) #31
  br label %44

44:                                               ; preds = %25, %34, %42
  %45 = phi ptr [ %40, %34 ], [ %43, %42 ], [ null, %25 ]
  store i64 %13, ptr %7, align 8
  store i64 %26, ptr %14, align 8
  %46 = icmp ugt i64 %15, 15
  br i1 %46, label %47, label %73

47:                                               ; preds = %44
  %48 = load ptr, ptr %0, align 8
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 1 %45, ptr align 1 %48, i64 %3, i1 false)
  %49 = getelementptr inbounds nuw i8, ptr %45, i64 %3
  tail call void @llvm.memset.p0.i64(ptr align 1 %49, i8 %5, i64 %4, i1 false)
  %50 = add i64 %8, 1
  %51 = sub i64 %50, %3
  %52 = getelementptr inbounds nuw i8, ptr %48, i64 %3
  %53 = getelementptr inbounds nuw i8, ptr %49, i64 %4
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 1 %53, ptr align 1 %52, i64 %51, i1 false)
  %54 = add i64 %15, 1
  %55 = icmp ugt i64 %54, 4095
  br i1 %55, label %56, label %70

56:                                               ; preds = %47
  %57 = getelementptr inbounds i8, ptr %48, i64 -8
  %58 = load i64, ptr %57, align 8
  %59 = ptrtoint ptr %48 to i64
  %60 = add i64 %59, -8
  %61 = sub i64 %60, %58
  %62 = icmp ult i64 %61, 32
  br i1 %62, label %65, label %63

63:                                               ; preds = %56
  invoke void @_invalid_parameter_noinfo_noreturn() #29
          to label %64 unwind label %68

64:                                               ; preds = %63
  unreachable

65:                                               ; preds = %56
  %66 = add i64 %15, 40
  %67 = inttoptr i64 %58 to ptr
  br label %70

68:                                               ; preds = %63
  %69 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %69) ]
  unreachable

70:                                               ; preds = %47, %65
  %71 = phi i64 [ %66, %65 ], [ %54, %47 ]
  %72 = phi ptr [ %67, %65 ], [ %48, %47 ]
  tail call void @"??3@YAXPEAX_K@Z"(ptr noundef %72, i64 noundef %71) #10
  br label %79

73:                                               ; preds = %44
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 1 %45, ptr nonnull align 8 %0, i64 %3, i1 false)
  %74 = getelementptr inbounds nuw i8, ptr %45, i64 %3
  tail call void @llvm.memset.p0.i64(ptr align 1 %74, i8 %5, i64 %4, i1 false)
  %75 = sub i64 %8, %3
  %76 = add i64 %75, 1
  %77 = getelementptr inbounds nuw i8, ptr %0, i64 %3
  %78 = getelementptr inbounds nuw i8, ptr %74, i64 %4
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 1 %78, ptr nonnull align 1 %77, i64 %76, i1 false)
  br label %79

79:                                               ; preds = %73, %70
  store ptr %45, ptr %0, align 8
  ret ptr %0
}

; Function Attrs: mustprogress noreturn uwtable
define linkonce_odr dso_local void @"?_Xran@?$_String_val@U?$_Simple_types@D@std@@@std@@SAXXZ"() local_unnamed_addr #20 comdat align 2 {
  tail call void @"?_Xout_of_range@std@@YAXPEBD@Z"(ptr noundef nonnull @"??_C@_0BI@CFPLBAOH@invalid?5string?5position?$AA@") #29
  unreachable
}

; Function Attrs: noreturn
declare dso_local void @"?_Xout_of_range@std@@YAXPEBD@Z"(ptr noundef) local_unnamed_addr #14

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_start.p0(ptr) #21

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_end.p0(ptr) #21

declare dso_local i32 @__stdio_common_vsprintf_s(i64 noundef, ptr noundef, i64 noundef, ptr noundef, ptr noundef, ptr noundef) local_unnamed_addr #11

; Function Attrs: mustprogress noinline nounwind uwtable
define linkonce_odr dso_local ptr @__local_stdio_printf_options() local_unnamed_addr #22 comdat {
  ret ptr @"?_OptionsStorage@?1??__local_stdio_printf_options@@9@4_KA"
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fabs.f64(double) #23

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.abs.i32(i32, i1 immarg) #23

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @"??$_Fput_v3@$0A@@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@1@V21@AEAVios_base@1@DPEBD_K_N@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr dead_on_unwind noalias writable sret(%"class.std::ostreambuf_iterator") align 8 %1, ptr dead_on_return noundef %2, ptr noundef nonnull align 8 dereferenceable(72) %3, i8 noundef %4, ptr noundef %5, i64 noundef %6, i1 noundef zeroext %7) local_unnamed_addr #2 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %9 = alloca [2 x i8], align 2
  %10 = alloca %"class.std::locale", align 8
  %11 = alloca %"class.std::basic_string", align 8
  %12 = alloca %"class.std::locale", align 8
  %13 = alloca %"class.std::basic_string", align 8
  %14 = alloca [7 x i8], align 1
  %15 = alloca [7 x i8], align 1
  %16 = alloca [7 x i8], align 1
  %17 = alloca [7 x i8], align 1
  %18 = alloca [7 x i8], align 1
  %19 = icmp eq i64 %6, 0
  br i1 %19, label %26, label %20

20:                                               ; preds = %8
  %21 = load i8, ptr %5, align 1
  %22 = icmp eq i8 %21, 43
  br i1 %22, label %26, label %23

23:                                               ; preds = %20
  %24 = icmp eq i8 %21, 45
  %25 = zext i1 %24 to i64
  br label %26

26:                                               ; preds = %20, %23, %8
  %27 = phi i64 [ 0, %8 ], [ 1, %20 ], [ %25, %23 ]
  %28 = getelementptr inbounds nuw i8, ptr %3, i64 24
  %29 = load i32, ptr %28, align 8
  %30 = and i32 %29, 12288
  %31 = icmp eq i32 %30, 12288
  br i1 %31, label %32, label %43

32:                                               ; preds = %26
  %33 = or disjoint i64 %27, 2
  %34 = icmp ugt i64 %33, %6
  br i1 %34, label %43, label %35

35:                                               ; preds = %32
  %36 = getelementptr inbounds nuw i8, ptr %5, i64 %27
  %37 = load i8, ptr %36, align 1
  %38 = icmp eq i8 %37, 48
  br i1 %38, label %39, label %43

39:                                               ; preds = %35
  %40 = getelementptr inbounds nuw i8, ptr %36, i64 1
  %41 = load i8, ptr %40, align 1
  switch i8 %41, label %43 [
    i8 120, label %42
    i8 88, label %42
  ]

42:                                               ; preds = %39, %39
  br label %43

43:                                               ; preds = %39, %26, %32, %35, %42
  %44 = phi ptr [ @"??_C@_02OOPEBDOJ@pP?$AA@", %42 ], [ @"??_C@_02OOPEBDOJ@pP?$AA@", %35 ], [ @"??_C@_02OOPEBDOJ@pP?$AA@", %32 ], [ @"??_C@_02MDKMJEGG@eE?$AA@", %26 ], [ @"??_C@_02OOPEBDOJ@pP?$AA@", %39 ]
  %45 = phi i64 [ %33, %42 ], [ %27, %35 ], [ %27, %32 ], [ %27, %26 ], [ %27, %39 ]
  %46 = tail call i64 @strcspn(ptr noundef %5, ptr noundef nonnull %44)
  call void @llvm.lifetime.start.p0(i64 2, ptr nonnull %9) #10
  store i16 46, ptr %9, align 2
  %47 = tail call ptr @localeconv()
  %48 = load ptr, ptr %47, align 8
  %49 = load i8, ptr %48, align 1
  store i8 %49, ptr %9, align 2
  %50 = call i64 @strcspn(ptr noundef %5, ptr noundef nonnull %9)
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %10) #10
  tail call void @llvm.experimental.noalias.scope.decl(metadata !93)
  %51 = getelementptr inbounds nuw i8, ptr %3, i64 64
  %52 = load ptr, ptr %51, align 8, !noalias !93
  %53 = getelementptr inbounds nuw i8, ptr %10, i64 8
  %54 = getelementptr inbounds nuw i8, ptr %52, i64 8
  %55 = load ptr, ptr %54, align 8, !noalias !93
  store ptr %55, ptr %53, align 8, !alias.scope !93
  %56 = load ptr, ptr %55, align 8, !noalias !93
  %57 = getelementptr inbounds nuw i8, ptr %56, i64 8
  %58 = load ptr, ptr %57, align 8, !noalias !93
  tail call void %58(ptr noundef nonnull align 8 dereferenceable(16) %55) #10, !noalias !93
  %59 = invoke noundef nonnull align 8 dereferenceable(48) ptr @"??$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z"(ptr noundef nonnull align 8 dereferenceable(16) %10)
          to label %60 unwind label %160

60:                                               ; preds = %43
  %61 = load ptr, ptr %53, align 8
  %62 = icmp eq ptr %61, null
  br i1 %62, label %73, label %63

63:                                               ; preds = %60
  %64 = load ptr, ptr %61, align 8
  %65 = getelementptr inbounds nuw i8, ptr %64, i64 16
  %66 = load ptr, ptr %65, align 8
  %67 = call noundef ptr %66(ptr noundef nonnull align 8 dereferenceable(16) %61) #10
  %68 = icmp eq ptr %67, null
  br i1 %68, label %73, label %69

69:                                               ; preds = %63
  %70 = load ptr, ptr %67, align 8
  %71 = load ptr, ptr %70, align 8
  %72 = call noundef ptr %71(ptr noundef nonnull align 8 dereferenceable(8) %67, i32 noundef 1) #10
  br label %73

73:                                               ; preds = %60, %63, %69
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %10) #10
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %11) #10
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %11, i8 0, i64 32, i1 false)
  %74 = icmp slt i64 %6, 0
  br i1 %74, label %75, label %76

75:                                               ; preds = %73
  call void @"?_Xlen_string@std@@YAXXZ"() #29
  unreachable

76:                                               ; preds = %73
  %77 = icmp ult i64 %6, 16
  br i1 %77, label %78, label %82

78:                                               ; preds = %76
  %79 = getelementptr inbounds nuw i8, ptr %11, i64 16
  store i64 %6, ptr %79, align 8
  %80 = getelementptr inbounds nuw i8, ptr %11, i64 24
  store i64 15, ptr %80, align 8
  call void @llvm.memset.p0.i64(ptr nonnull align 8 dereferenceable(32) %11, i8 0, i64 %6, i1 false)
  %81 = getelementptr inbounds nuw [16 x i8], ptr %11, i64 0, i64 %6
  store i8 0, ptr %81, align 1
  br label %105

82:                                               ; preds = %76
  %83 = or i64 %6, 15
  %84 = call i64 @llvm.umax.i64(i64 %83, i64 22)
  %85 = icmp ugt i64 %83, 4094
  br i1 %85, label %86, label %97

86:                                               ; preds = %82
  %87 = icmp ult i64 %83, -40
  br i1 %87, label %89, label %88

88:                                               ; preds = %86
  call void @"?_Throw_bad_array_new_length@std@@YAXXZ"() #29
  unreachable

89:                                               ; preds = %86
  %90 = add nuw i64 %84, 40
  %91 = call noalias noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef %90) #31
  %92 = ptrtoint ptr %91 to i64
  %93 = add i64 %92, 39
  %94 = and i64 %93, -32
  %95 = inttoptr i64 %94 to ptr
  %96 = getelementptr inbounds i8, ptr %95, i64 -8
  store i64 %92, ptr %96, align 8
  br label %100

97:                                               ; preds = %82
  %98 = add nuw nsw i64 %84, 1
  %99 = call noalias noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef %98) #31
  br label %100

100:                                              ; preds = %97, %89
  %101 = phi ptr [ %95, %89 ], [ %99, %97 ]
  store ptr %101, ptr %11, align 8
  %102 = getelementptr inbounds nuw i8, ptr %11, i64 16
  store i64 %6, ptr %102, align 8
  %103 = getelementptr inbounds nuw i8, ptr %11, i64 24
  store i64 %84, ptr %103, align 8
  call void @llvm.memset.p0.i64(ptr align 1 %101, i8 0, i64 %6, i1 false)
  %104 = getelementptr inbounds nuw i8, ptr %101, i64 %6
  store i8 0, ptr %104, align 1
  br label %105

105:                                              ; preds = %78, %100
  %106 = getelementptr inbounds nuw i8, ptr %11, i64 24
  %107 = load i64, ptr %106, align 8
  %108 = icmp ugt i64 %107, 15
  %109 = load ptr, ptr %11, align 8
  %110 = select i1 %108, ptr %109, ptr %11
  %111 = getelementptr inbounds nuw i8, ptr %5, i64 %6
  %112 = load ptr, ptr %59, align 8
  %113 = getelementptr inbounds nuw i8, ptr %112, i64 56
  %114 = load ptr, ptr %113, align 8
  %115 = invoke noundef ptr %114(ptr noundef nonnull align 8 dereferenceable(48) %59, ptr noundef %5, ptr noundef %111, ptr noundef nonnull %110)
          to label %116 unwind label %605

116:                                              ; preds = %105
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %12) #10
  call void @llvm.experimental.noalias.scope.decl(metadata !96)
  %117 = load ptr, ptr %51, align 8, !noalias !96
  %118 = getelementptr inbounds nuw i8, ptr %12, i64 8
  %119 = getelementptr inbounds nuw i8, ptr %117, i64 8
  %120 = load ptr, ptr %119, align 8, !noalias !96
  store ptr %120, ptr %118, align 8, !alias.scope !96
  %121 = load ptr, ptr %120, align 8, !noalias !96
  %122 = getelementptr inbounds nuw i8, ptr %121, i64 8
  %123 = load ptr, ptr %122, align 8, !noalias !96
  call void %123(ptr noundef nonnull align 8 dereferenceable(16) %120) #10, !noalias !96
  %124 = invoke noundef nonnull align 8 dereferenceable(48) ptr @"??$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z"(ptr noundef nonnull align 8 dereferenceable(16) %12)
          to label %125 unwind label %175

125:                                              ; preds = %116
  %126 = load ptr, ptr %118, align 8
  %127 = icmp eq ptr %126, null
  br i1 %127, label %138, label %128

128:                                              ; preds = %125
  %129 = load ptr, ptr %126, align 8
  %130 = getelementptr inbounds nuw i8, ptr %129, i64 16
  %131 = load ptr, ptr %130, align 8
  %132 = call noundef ptr %131(ptr noundef nonnull align 8 dereferenceable(16) %126) #10
  %133 = icmp eq ptr %132, null
  br i1 %133, label %138, label %134

134:                                              ; preds = %128
  %135 = load ptr, ptr %132, align 8
  %136 = load ptr, ptr %135, align 8
  %137 = call noundef ptr %136(ptr noundef nonnull align 8 dereferenceable(8) %132, i32 noundef 1) #10
  br label %138

138:                                              ; preds = %125, %128, %134
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %12) #10
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %13) #10
  %139 = load ptr, ptr %124, align 8, !noalias !99
  %140 = getelementptr inbounds nuw i8, ptr %139, i64 40
  %141 = load ptr, ptr %140, align 8, !noalias !99
  invoke void %141(ptr noundef nonnull align 8 dereferenceable(48) %124, ptr dead_on_unwind nonnull writable sret(%"class.std::basic_string") align 8 %13)
          to label %142 unwind label %605

142:                                              ; preds = %138
  %143 = load ptr, ptr %124, align 8
  %144 = getelementptr inbounds nuw i8, ptr %143, i64 32
  %145 = load ptr, ptr %144, align 8
  %146 = invoke noundef i8 %145(ptr noundef nonnull align 8 dereferenceable(48) %124)
          to label %147 unwind label %603

147:                                              ; preds = %142
  %148 = icmp eq i64 %50, %6
  br i1 %148, label %190, label %149

149:                                              ; preds = %147
  %150 = load ptr, ptr %124, align 8
  %151 = getelementptr inbounds nuw i8, ptr %150, i64 24
  %152 = load ptr, ptr %151, align 8
  %153 = invoke noundef i8 %152(ptr noundef nonnull align 8 dereferenceable(48) %124)
          to label %154 unwind label %603

154:                                              ; preds = %149
  %155 = load i64, ptr %106, align 8
  %156 = icmp ugt i64 %155, 15
  %157 = load ptr, ptr %11, align 8
  %158 = select i1 %156, ptr %157, ptr %11
  %159 = getelementptr inbounds nuw i8, ptr %158, i64 %50
  store i8 %153, ptr %159, align 1
  br label %190

160:                                              ; preds = %43
  %161 = cleanuppad within none []
  %162 = load ptr, ptr %53, align 8
  %163 = icmp eq ptr %162, null
  br i1 %163, label %174, label %164

164:                                              ; preds = %160
  %165 = load ptr, ptr %162, align 8
  %166 = getelementptr inbounds nuw i8, ptr %165, i64 16
  %167 = load ptr, ptr %166, align 8
  %168 = call noundef ptr %167(ptr noundef nonnull align 8 dereferenceable(16) %162) #10 [ "funclet"(token %161) ]
  %169 = icmp eq ptr %168, null
  br i1 %169, label %174, label %170

170:                                              ; preds = %164
  %171 = load ptr, ptr %168, align 8
  %172 = load ptr, ptr %171, align 8
  %173 = call noundef ptr %172(ptr noundef nonnull align 8 dereferenceable(8) %168, i32 noundef 1) #10 [ "funclet"(token %161) ]
  br label %174

174:                                              ; preds = %160, %164, %170
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %10) #10
  cleanupret from %161 unwind to caller

175:                                              ; preds = %116
  %176 = cleanuppad within none []
  %177 = load ptr, ptr %118, align 8
  %178 = icmp eq ptr %177, null
  br i1 %178, label %189, label %179

179:                                              ; preds = %175
  %180 = load ptr, ptr %177, align 8
  %181 = getelementptr inbounds nuw i8, ptr %180, i64 16
  %182 = load ptr, ptr %181, align 8
  %183 = call noundef ptr %182(ptr noundef nonnull align 8 dereferenceable(16) %177) #10 [ "funclet"(token %176) ]
  %184 = icmp eq ptr %183, null
  br i1 %184, label %189, label %185

185:                                              ; preds = %179
  %186 = load ptr, ptr %183, align 8
  %187 = load ptr, ptr %186, align 8
  %188 = call noundef ptr %187(ptr noundef nonnull align 8 dereferenceable(8) %183, i32 noundef 1) #10 [ "funclet"(token %176) ]
  br label %189

189:                                              ; preds = %175, %179, %185
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %12) #10
  cleanupret from %176 unwind label %605

190:                                              ; preds = %154, %147
  br i1 %7, label %191, label %234

191:                                              ; preds = %190
  %192 = select i1 %148, i64 %46, i64 %50
  %193 = getelementptr inbounds nuw i8, ptr %13, i64 24
  %194 = load i64, ptr %193, align 8
  %195 = icmp ugt i64 %194, 15
  %196 = load ptr, ptr %13, align 8
  %197 = select i1 %195, ptr %196, ptr %13
  %198 = getelementptr inbounds nuw i8, ptr %11, i64 16
  br label %199

199:                                              ; preds = %229, %191
  %200 = phi i64 [ %192, %191 ], [ %210, %229 ]
  %201 = phi ptr [ %197, %191 ], [ %233, %229 ]
  %202 = load i8, ptr %201, align 1
  %203 = add i8 %202, -1
  %204 = icmp ult i8 %203, 126
  br i1 %204, label %205, label %234

205:                                              ; preds = %199
  %206 = zext nneg i8 %202 to i64
  %207 = sub i64 %200, %45
  %208 = icmp ugt i64 %207, %206
  br i1 %208, label %209, label %234

209:                                              ; preds = %205
  %210 = sub i64 %200, %206
  %211 = load i64, ptr %198, align 8
  %212 = icmp ult i64 %211, %210
  br i1 %212, label %213, label %215

213:                                              ; preds = %209
  invoke void @"?_Xran@?$_String_val@U?$_Simple_types@D@std@@@std@@SAXXZ"() #29
          to label %214 unwind label %603

214:                                              ; preds = %213
  unreachable

215:                                              ; preds = %209
  %216 = load i64, ptr %106, align 8
  %217 = icmp eq i64 %216, %211
  br i1 %217, label %227, label %218

218:                                              ; preds = %215
  %219 = add i64 %211, 1
  store i64 %219, ptr %198, align 8
  %220 = icmp ugt i64 %216, 15
  %221 = load ptr, ptr %11, align 8
  %222 = select i1 %220, ptr %221, ptr %11
  %223 = getelementptr inbounds nuw i8, ptr %222, i64 %210
  %224 = sub i64 %211, %210
  %225 = add i64 %224, 1
  %226 = getelementptr inbounds nuw i8, ptr %223, i64 1
  call void @llvm.memmove.p0.p0.i64(ptr nonnull align 1 %226, ptr align 1 %223, i64 %225, i1 false)
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(1) %223, i8 %146, i64 1, i1 false)
  br label %229

227:                                              ; preds = %215
  %228 = invoke noundef nonnull align 8 dereferenceable(32) ptr @"??$_Reallocate_grow_by@V<lambda_1>@?0??insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@_K0D@Z@_K_KD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??insert@01@QEAAAEAV01@00D@Z@_K2D@Z"(ptr noundef nonnull align 8 dereferenceable(32) %11, i64 noundef 1, i8 undef, i64 noundef %210, i64 noundef 1, i8 noundef %146)
          to label %229 unwind label %603

229:                                              ; preds = %218, %227
  %230 = getelementptr inbounds nuw i8, ptr %201, i64 1
  %231 = load i8, ptr %230, align 1
  %232 = icmp sgt i8 %231, 0
  %233 = select i1 %232, ptr %230, ptr %201
  br label %199, !llvm.loop !102

234:                                              ; preds = %205, %199, %190
  %235 = getelementptr inbounds nuw i8, ptr %11, i64 16
  %236 = load i64, ptr %235, align 8
  %237 = getelementptr inbounds nuw i8, ptr %3, i64 40
  %238 = load i64, ptr %237, align 8
  %239 = icmp sgt i64 %238, 0
  %240 = icmp ugt i64 %238, %236
  %241 = select i1 %239, i1 %240, i1 false
  %242 = sub i64 %238, %236
  %243 = select i1 %241, i64 %242, i64 0
  %244 = load i32, ptr %28, align 8
  %245 = and i32 %244, 448
  switch i32 %245, label %246 [
    i32 256, label %330
    i32 64, label %414
  ]

246:                                              ; preds = %234
  %247 = load i8, ptr %2, align 8
  %248 = getelementptr inbounds nuw i8, ptr %2, i64 1
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %14, ptr noundef nonnull align 1 dereferenceable(7) %248, i64 7, i1 false)
  %249 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %250 = load ptr, ptr %249, align 8
  %251 = icmp eq i64 %243, 0
  br i1 %251, label %285, label %252

252:                                              ; preds = %246
  %253 = zext i8 %4 to i32
  %254 = icmp eq ptr %250, null
  %255 = getelementptr inbounds nuw i8, ptr %250, i64 64
  %256 = getelementptr inbounds nuw i8, ptr %250, i64 88
  br label %257

257:                                              ; preds = %281, %252
  %258 = phi i8 [ %247, %252 ], [ %282, %281 ]
  %259 = phi i64 [ %243, %252 ], [ %283, %281 ]
  br i1 %254, label %280, label %260

260:                                              ; preds = %257
  %261 = load ptr, ptr %255, align 8, !noalias !103
  %262 = load ptr, ptr %261, align 8, !noalias !103
  %263 = icmp eq ptr %262, null
  br i1 %263, label %273, label %264

264:                                              ; preds = %260
  %265 = load ptr, ptr %256, align 8, !noalias !103
  %266 = load i32, ptr %265, align 4, !noalias !103
  %267 = icmp sgt i32 %266, 0
  br i1 %267, label %268, label %273

268:                                              ; preds = %264
  %269 = add nsw i32 %266, -1
  store i32 %269, ptr %265, align 4, !noalias !103
  %270 = load ptr, ptr %255, align 8, !noalias !103
  %271 = load ptr, ptr %270, align 8, !noalias !103
  %272 = getelementptr inbounds nuw i8, ptr %271, i64 1
  store ptr %272, ptr %270, align 8, !noalias !103
  store i8 %4, ptr %271, align 1, !noalias !103
  br label %281

273:                                              ; preds = %264, %260
  %274 = load ptr, ptr %250, align 8, !noalias !103
  %275 = getelementptr inbounds nuw i8, ptr %274, i64 24
  %276 = load ptr, ptr %275, align 8, !noalias !103
  %277 = invoke noundef i32 %276(ptr noundef nonnull align 8 dereferenceable(104) %250, i32 noundef %253)
          to label %278 unwind label %603

278:                                              ; preds = %273
  %279 = icmp eq i32 %277, -1
  br i1 %279, label %280, label %281

280:                                              ; preds = %278, %257
  br label %281

281:                                              ; preds = %280, %278, %268
  %282 = phi i8 [ 1, %280 ], [ %258, %278 ], [ %258, %268 ]
  %283 = add i64 %259, -1
  %284 = icmp eq i64 %283, 0
  br i1 %284, label %285, label %257, !llvm.loop !54

285:                                              ; preds = %281, %246
  %286 = phi i8 [ %247, %246 ], [ %282, %281 ]
  store i8 %286, ptr %2, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %248, ptr noundef nonnull align 1 dereferenceable(7) %14, i64 7, i1 false)
  store ptr %250, ptr %249, align 8
  %287 = icmp eq i64 %45, 0
  br i1 %287, label %328, label %288

288:                                              ; preds = %285
  %289 = load ptr, ptr %11, align 8
  %290 = load i64, ptr %106, align 8
  %291 = icmp ugt i64 %290, 15
  %292 = select i1 %291, ptr %289, ptr %11
  %293 = icmp eq ptr %250, null
  %294 = getelementptr inbounds nuw i8, ptr %250, i64 64
  %295 = getelementptr inbounds nuw i8, ptr %250, i64 88
  br label %296

296:                                              ; preds = %288, %323
  %297 = phi i8 [ %324, %323 ], [ %286, %288 ]
  %298 = phi i64 [ %325, %323 ], [ %45, %288 ]
  %299 = phi ptr [ %326, %323 ], [ %292, %288 ]
  %300 = load i8, ptr %299, align 1, !noalias !106
  br i1 %293, label %322, label %301

301:                                              ; preds = %296
  %302 = load ptr, ptr %294, align 8, !noalias !106
  %303 = load ptr, ptr %302, align 8, !noalias !106
  %304 = icmp eq ptr %303, null
  br i1 %304, label %314, label %305

305:                                              ; preds = %301
  %306 = load ptr, ptr %295, align 8, !noalias !106
  %307 = load i32, ptr %306, align 4, !noalias !106
  %308 = icmp sgt i32 %307, 0
  br i1 %308, label %309, label %314

309:                                              ; preds = %305
  %310 = add nsw i32 %307, -1
  store i32 %310, ptr %306, align 4, !noalias !106
  %311 = load ptr, ptr %294, align 8, !noalias !106
  %312 = load ptr, ptr %311, align 8, !noalias !106
  %313 = getelementptr inbounds nuw i8, ptr %312, i64 1
  store ptr %313, ptr %311, align 8, !noalias !106
  store i8 %300, ptr %312, align 1, !noalias !106
  br label %323

314:                                              ; preds = %305, %301
  %315 = zext i8 %300 to i32
  %316 = load ptr, ptr %250, align 8, !noalias !106
  %317 = getelementptr inbounds nuw i8, ptr %316, i64 24
  %318 = load ptr, ptr %317, align 8, !noalias !106
  %319 = invoke noundef i32 %318(ptr noundef nonnull align 8 dereferenceable(104) %250, i32 noundef %315)
          to label %320 unwind label %603

320:                                              ; preds = %314
  %321 = icmp eq i32 %319, -1
  br i1 %321, label %322, label %323

322:                                              ; preds = %320, %296
  br label %323

323:                                              ; preds = %322, %320, %309
  %324 = phi i8 [ 1, %322 ], [ %297, %320 ], [ %297, %309 ]
  %325 = add i64 %298, -1
  %326 = getelementptr inbounds nuw i8, ptr %299, i64 1
  %327 = icmp eq i64 %325, 0
  br i1 %327, label %328, label %296, !llvm.loop !58

328:                                              ; preds = %323, %285
  %329 = phi i8 [ %286, %285 ], [ %324, %323 ]
  store i8 %329, ptr %2, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %248, ptr noundef nonnull align 1 dereferenceable(7) %14, i64 7, i1 false)
  store ptr %250, ptr %249, align 8
  br label %462

330:                                              ; preds = %234
  %331 = load i64, ptr %106, align 8
  %332 = load ptr, ptr %11, align 8
  %333 = load i8, ptr %2, align 8
  %334 = getelementptr inbounds nuw i8, ptr %2, i64 1
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %15, ptr noundef nonnull align 1 dereferenceable(7) %334, i64 7, i1 false)
  %335 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %336 = load ptr, ptr %335, align 8
  %337 = icmp eq i64 %45, 0
  br i1 %337, label %376, label %338

338:                                              ; preds = %330
  %339 = icmp ugt i64 %331, 15
  %340 = select i1 %339, ptr %332, ptr %11
  %341 = icmp eq ptr %336, null
  %342 = getelementptr inbounds nuw i8, ptr %336, i64 64
  %343 = getelementptr inbounds nuw i8, ptr %336, i64 88
  br label %344

344:                                              ; preds = %338, %371
  %345 = phi i8 [ %372, %371 ], [ %333, %338 ]
  %346 = phi i64 [ %373, %371 ], [ %45, %338 ]
  %347 = phi ptr [ %374, %371 ], [ %340, %338 ]
  %348 = load i8, ptr %347, align 1, !noalias !109
  br i1 %341, label %370, label %349

349:                                              ; preds = %344
  %350 = load ptr, ptr %342, align 8, !noalias !109
  %351 = load ptr, ptr %350, align 8, !noalias !109
  %352 = icmp eq ptr %351, null
  br i1 %352, label %362, label %353

353:                                              ; preds = %349
  %354 = load ptr, ptr %343, align 8, !noalias !109
  %355 = load i32, ptr %354, align 4, !noalias !109
  %356 = icmp sgt i32 %355, 0
  br i1 %356, label %357, label %362

357:                                              ; preds = %353
  %358 = add nsw i32 %355, -1
  store i32 %358, ptr %354, align 4, !noalias !109
  %359 = load ptr, ptr %342, align 8, !noalias !109
  %360 = load ptr, ptr %359, align 8, !noalias !109
  %361 = getelementptr inbounds nuw i8, ptr %360, i64 1
  store ptr %361, ptr %359, align 8, !noalias !109
  store i8 %348, ptr %360, align 1, !noalias !109
  br label %371

362:                                              ; preds = %353, %349
  %363 = zext i8 %348 to i32
  %364 = load ptr, ptr %336, align 8, !noalias !109
  %365 = getelementptr inbounds nuw i8, ptr %364, i64 24
  %366 = load ptr, ptr %365, align 8, !noalias !109
  %367 = invoke noundef i32 %366(ptr noundef nonnull align 8 dereferenceable(104) %336, i32 noundef %363)
          to label %368 unwind label %603

368:                                              ; preds = %362
  %369 = icmp eq i32 %367, -1
  br i1 %369, label %370, label %371

370:                                              ; preds = %368, %344
  br label %371

371:                                              ; preds = %370, %368, %357
  %372 = phi i8 [ 1, %370 ], [ %345, %368 ], [ %345, %357 ]
  %373 = add i64 %346, -1
  %374 = getelementptr inbounds nuw i8, ptr %347, i64 1
  %375 = icmp eq i64 %373, 0
  br i1 %375, label %376, label %344, !llvm.loop !58

376:                                              ; preds = %371, %330
  %377 = phi i8 [ %333, %330 ], [ %372, %371 ]
  store i8 %377, ptr %2, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %334, ptr noundef nonnull align 1 dereferenceable(7) %15, i64 7, i1 false)
  store ptr %336, ptr %335, align 8
  %378 = icmp eq i64 %243, 0
  br i1 %378, label %412, label %379

379:                                              ; preds = %376
  %380 = zext i8 %4 to i32
  %381 = icmp eq ptr %336, null
  %382 = getelementptr inbounds nuw i8, ptr %336, i64 64
  %383 = getelementptr inbounds nuw i8, ptr %336, i64 88
  br label %384

384:                                              ; preds = %408, %379
  %385 = phi i8 [ %377, %379 ], [ %409, %408 ]
  %386 = phi i64 [ %243, %379 ], [ %410, %408 ]
  br i1 %381, label %407, label %387

387:                                              ; preds = %384
  %388 = load ptr, ptr %382, align 8, !noalias !112
  %389 = load ptr, ptr %388, align 8, !noalias !112
  %390 = icmp eq ptr %389, null
  br i1 %390, label %400, label %391

391:                                              ; preds = %387
  %392 = load ptr, ptr %383, align 8, !noalias !112
  %393 = load i32, ptr %392, align 4, !noalias !112
  %394 = icmp sgt i32 %393, 0
  br i1 %394, label %395, label %400

395:                                              ; preds = %391
  %396 = add nsw i32 %393, -1
  store i32 %396, ptr %392, align 4, !noalias !112
  %397 = load ptr, ptr %382, align 8, !noalias !112
  %398 = load ptr, ptr %397, align 8, !noalias !112
  %399 = getelementptr inbounds nuw i8, ptr %398, i64 1
  store ptr %399, ptr %397, align 8, !noalias !112
  store i8 %4, ptr %398, align 1, !noalias !112
  br label %408

400:                                              ; preds = %391, %387
  %401 = load ptr, ptr %336, align 8, !noalias !112
  %402 = getelementptr inbounds nuw i8, ptr %401, i64 24
  %403 = load ptr, ptr %402, align 8, !noalias !112
  %404 = invoke noundef i32 %403(ptr noundef nonnull align 8 dereferenceable(104) %336, i32 noundef %380)
          to label %405 unwind label %603

405:                                              ; preds = %400
  %406 = icmp eq i32 %404, -1
  br i1 %406, label %407, label %408

407:                                              ; preds = %405, %384
  br label %408

408:                                              ; preds = %407, %405, %395
  %409 = phi i8 [ 1, %407 ], [ %385, %405 ], [ %385, %395 ]
  %410 = add i64 %386, -1
  %411 = icmp eq i64 %410, 0
  br i1 %411, label %412, label %384, !llvm.loop !54

412:                                              ; preds = %408, %376
  %413 = phi i8 [ %377, %376 ], [ %409, %408 ]
  store i8 %413, ptr %2, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %334, ptr noundef nonnull align 1 dereferenceable(7) %15, i64 7, i1 false)
  store ptr %336, ptr %335, align 8
  br label %462

414:                                              ; preds = %234
  %415 = load i64, ptr %106, align 8
  %416 = load ptr, ptr %11, align 8
  %417 = load i8, ptr %2, align 8
  %418 = getelementptr inbounds nuw i8, ptr %2, i64 1
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %16, ptr noundef nonnull align 1 dereferenceable(7) %418, i64 7, i1 false)
  %419 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %420 = load ptr, ptr %419, align 8
  %421 = icmp eq i64 %45, 0
  br i1 %421, label %460, label %422

422:                                              ; preds = %414
  %423 = icmp ugt i64 %415, 15
  %424 = select i1 %423, ptr %416, ptr %11
  %425 = icmp eq ptr %420, null
  %426 = getelementptr inbounds nuw i8, ptr %420, i64 64
  %427 = getelementptr inbounds nuw i8, ptr %420, i64 88
  br label %428

428:                                              ; preds = %422, %455
  %429 = phi i8 [ %456, %455 ], [ %417, %422 ]
  %430 = phi i64 [ %457, %455 ], [ %45, %422 ]
  %431 = phi ptr [ %458, %455 ], [ %424, %422 ]
  %432 = load i8, ptr %431, align 1, !noalias !115
  br i1 %425, label %454, label %433

433:                                              ; preds = %428
  %434 = load ptr, ptr %426, align 8, !noalias !115
  %435 = load ptr, ptr %434, align 8, !noalias !115
  %436 = icmp eq ptr %435, null
  br i1 %436, label %446, label %437

437:                                              ; preds = %433
  %438 = load ptr, ptr %427, align 8, !noalias !115
  %439 = load i32, ptr %438, align 4, !noalias !115
  %440 = icmp sgt i32 %439, 0
  br i1 %440, label %441, label %446

441:                                              ; preds = %437
  %442 = add nsw i32 %439, -1
  store i32 %442, ptr %438, align 4, !noalias !115
  %443 = load ptr, ptr %426, align 8, !noalias !115
  %444 = load ptr, ptr %443, align 8, !noalias !115
  %445 = getelementptr inbounds nuw i8, ptr %444, i64 1
  store ptr %445, ptr %443, align 8, !noalias !115
  store i8 %432, ptr %444, align 1, !noalias !115
  br label %455

446:                                              ; preds = %437, %433
  %447 = zext i8 %432 to i32
  %448 = load ptr, ptr %420, align 8, !noalias !115
  %449 = getelementptr inbounds nuw i8, ptr %448, i64 24
  %450 = load ptr, ptr %449, align 8, !noalias !115
  %451 = invoke noundef i32 %450(ptr noundef nonnull align 8 dereferenceable(104) %420, i32 noundef %447)
          to label %452 unwind label %603

452:                                              ; preds = %446
  %453 = icmp eq i32 %451, -1
  br i1 %453, label %454, label %455

454:                                              ; preds = %452, %428
  br label %455

455:                                              ; preds = %454, %452, %441
  %456 = phi i8 [ 1, %454 ], [ %429, %452 ], [ %429, %441 ]
  %457 = add i64 %430, -1
  %458 = getelementptr inbounds nuw i8, ptr %431, i64 1
  %459 = icmp eq i64 %457, 0
  br i1 %459, label %460, label %428, !llvm.loop !58

460:                                              ; preds = %455, %414
  %461 = phi i8 [ %417, %414 ], [ %456, %455 ]
  store i8 %461, ptr %2, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %418, ptr noundef nonnull align 1 dereferenceable(7) %16, i64 7, i1 false)
  store ptr %420, ptr %419, align 8
  br label %462

462:                                              ; preds = %412, %460, %328
  %463 = phi i64 [ 0, %328 ], [ 0, %412 ], [ %243, %460 ]
  %464 = sub i64 %236, %45
  %465 = load i64, ptr %106, align 8
  %466 = load ptr, ptr %11, align 8
  %467 = load i8, ptr %2, align 8
  %468 = getelementptr inbounds nuw i8, ptr %2, i64 1
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %17, ptr noundef nonnull align 1 dereferenceable(7) %468, i64 7, i1 false)
  %469 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %470 = load ptr, ptr %469, align 8
  %471 = icmp eq i64 %464, 0
  br i1 %471, label %511, label %472

472:                                              ; preds = %462
  %473 = icmp ugt i64 %465, 15
  %474 = select i1 %473, ptr %466, ptr %11
  %475 = getelementptr inbounds nuw i8, ptr %474, i64 %45
  %476 = icmp eq ptr %470, null
  %477 = getelementptr inbounds nuw i8, ptr %470, i64 64
  %478 = getelementptr inbounds nuw i8, ptr %470, i64 88
  br label %479

479:                                              ; preds = %472, %506
  %480 = phi i8 [ %507, %506 ], [ %467, %472 ]
  %481 = phi i64 [ %508, %506 ], [ %464, %472 ]
  %482 = phi ptr [ %509, %506 ], [ %475, %472 ]
  %483 = load i8, ptr %482, align 1, !noalias !118
  br i1 %476, label %505, label %484

484:                                              ; preds = %479
  %485 = load ptr, ptr %477, align 8, !noalias !118
  %486 = load ptr, ptr %485, align 8, !noalias !118
  %487 = icmp eq ptr %486, null
  br i1 %487, label %497, label %488

488:                                              ; preds = %484
  %489 = load ptr, ptr %478, align 8, !noalias !118
  %490 = load i32, ptr %489, align 4, !noalias !118
  %491 = icmp sgt i32 %490, 0
  br i1 %491, label %492, label %497

492:                                              ; preds = %488
  %493 = add nsw i32 %490, -1
  store i32 %493, ptr %489, align 4, !noalias !118
  %494 = load ptr, ptr %477, align 8, !noalias !118
  %495 = load ptr, ptr %494, align 8, !noalias !118
  %496 = getelementptr inbounds nuw i8, ptr %495, i64 1
  store ptr %496, ptr %494, align 8, !noalias !118
  store i8 %483, ptr %495, align 1, !noalias !118
  br label %506

497:                                              ; preds = %488, %484
  %498 = zext i8 %483 to i32
  %499 = load ptr, ptr %470, align 8, !noalias !118
  %500 = getelementptr inbounds nuw i8, ptr %499, i64 24
  %501 = load ptr, ptr %500, align 8, !noalias !118
  %502 = invoke noundef i32 %501(ptr noundef nonnull align 8 dereferenceable(104) %470, i32 noundef %498)
          to label %503 unwind label %603

503:                                              ; preds = %497
  %504 = icmp eq i32 %502, -1
  br i1 %504, label %505, label %506

505:                                              ; preds = %503, %479
  br label %506

506:                                              ; preds = %505, %503, %492
  %507 = phi i8 [ 1, %505 ], [ %480, %503 ], [ %480, %492 ]
  %508 = add i64 %481, -1
  %509 = getelementptr inbounds nuw i8, ptr %482, i64 1
  %510 = icmp eq i64 %508, 0
  br i1 %510, label %511, label %479, !llvm.loop !58

511:                                              ; preds = %506, %462
  %512 = phi i8 [ %467, %462 ], [ %507, %506 ]
  store i8 %512, ptr %2, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %468, ptr noundef nonnull align 1 dereferenceable(7) %17, i64 7, i1 false)
  store ptr %470, ptr %469, align 8
  store i64 0, ptr %237, align 8
  %513 = load i8, ptr %2, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %18, ptr noundef nonnull align 1 dereferenceable(7) %468, i64 7, i1 false)
  %514 = load ptr, ptr %469, align 8
  %515 = icmp eq i64 %463, 0
  br i1 %515, label %549, label %516

516:                                              ; preds = %511
  %517 = zext i8 %4 to i32
  %518 = icmp eq ptr %514, null
  %519 = getelementptr inbounds nuw i8, ptr %514, i64 64
  %520 = getelementptr inbounds nuw i8, ptr %514, i64 88
  br label %521

521:                                              ; preds = %545, %516
  %522 = phi i8 [ %513, %516 ], [ %546, %545 ]
  %523 = phi i64 [ %463, %516 ], [ %547, %545 ]
  br i1 %518, label %544, label %524

524:                                              ; preds = %521
  %525 = load ptr, ptr %519, align 8, !noalias !121
  %526 = load ptr, ptr %525, align 8, !noalias !121
  %527 = icmp eq ptr %526, null
  br i1 %527, label %537, label %528

528:                                              ; preds = %524
  %529 = load ptr, ptr %520, align 8, !noalias !121
  %530 = load i32, ptr %529, align 4, !noalias !121
  %531 = icmp sgt i32 %530, 0
  br i1 %531, label %532, label %537

532:                                              ; preds = %528
  %533 = add nsw i32 %530, -1
  store i32 %533, ptr %529, align 4, !noalias !121
  %534 = load ptr, ptr %519, align 8, !noalias !121
  %535 = load ptr, ptr %534, align 8, !noalias !121
  %536 = getelementptr inbounds nuw i8, ptr %535, i64 1
  store ptr %536, ptr %534, align 8, !noalias !121
  store i8 %4, ptr %535, align 1, !noalias !121
  br label %545

537:                                              ; preds = %528, %524
  %538 = load ptr, ptr %514, align 8, !noalias !121
  %539 = getelementptr inbounds nuw i8, ptr %538, i64 24
  %540 = load ptr, ptr %539, align 8, !noalias !121
  %541 = invoke noundef i32 %540(ptr noundef nonnull align 8 dereferenceable(104) %514, i32 noundef %517)
          to label %542 unwind label %603

542:                                              ; preds = %537
  %543 = icmp eq i32 %541, -1
  br i1 %543, label %544, label %545

544:                                              ; preds = %542, %521
  br label %545

545:                                              ; preds = %544, %542, %532
  %546 = phi i8 [ 1, %544 ], [ %522, %542 ], [ %522, %532 ]
  %547 = add i64 %523, -1
  %548 = icmp eq i64 %547, 0
  br i1 %548, label %549, label %521, !llvm.loop !54

549:                                              ; preds = %545, %511
  %550 = phi i8 [ %513, %511 ], [ %546, %545 ]
  store i8 %550, ptr %1, align 8
  %551 = getelementptr inbounds nuw i8, ptr %1, i64 1
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %551, ptr noundef nonnull align 1 dereferenceable(7) %18, i64 7, i1 false)
  %552 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store ptr %514, ptr %552, align 8
  %553 = getelementptr inbounds nuw i8, ptr %13, i64 24
  %554 = load i64, ptr %553, align 8
  %555 = icmp ugt i64 %554, 15
  br i1 %555, label %556, label %577

556:                                              ; preds = %549
  %557 = load ptr, ptr %13, align 8
  %558 = add i64 %554, 1
  %559 = icmp ugt i64 %558, 4095
  br i1 %559, label %560, label %574

560:                                              ; preds = %556
  %561 = getelementptr inbounds i8, ptr %557, i64 -8
  %562 = load i64, ptr %561, align 8
  %563 = ptrtoint ptr %557 to i64
  %564 = add i64 %563, -8
  %565 = sub i64 %564, %562
  %566 = icmp ult i64 %565, 32
  br i1 %566, label %569, label %567

567:                                              ; preds = %560
  invoke void @_invalid_parameter_noinfo_noreturn() #29
          to label %568 unwind label %572

568:                                              ; preds = %567
  unreachable

569:                                              ; preds = %560
  %570 = add i64 %554, 40
  %571 = inttoptr i64 %562 to ptr
  br label %574

572:                                              ; preds = %567
  %573 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %573) ]
  unreachable

574:                                              ; preds = %569, %556
  %575 = phi i64 [ %570, %569 ], [ %558, %556 ]
  %576 = phi ptr [ %571, %569 ], [ %557, %556 ]
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %576, i64 noundef %575) #10
  br label %577

577:                                              ; preds = %549, %574
  %578 = getelementptr inbounds nuw i8, ptr %13, i64 16
  store i64 0, ptr %578, align 8
  store i64 15, ptr %553, align 8
  store i8 0, ptr %13, align 8
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %13) #10
  %579 = load i64, ptr %106, align 8
  %580 = icmp ugt i64 %579, 15
  br i1 %580, label %581, label %602

581:                                              ; preds = %577
  %582 = load ptr, ptr %11, align 8
  %583 = add i64 %579, 1
  %584 = icmp ugt i64 %583, 4095
  br i1 %584, label %585, label %599

585:                                              ; preds = %581
  %586 = getelementptr inbounds i8, ptr %582, i64 -8
  %587 = load i64, ptr %586, align 8
  %588 = ptrtoint ptr %582 to i64
  %589 = add i64 %588, -8
  %590 = sub i64 %589, %587
  %591 = icmp ult i64 %590, 32
  br i1 %591, label %594, label %592

592:                                              ; preds = %585
  invoke void @_invalid_parameter_noinfo_noreturn() #29
          to label %593 unwind label %597

593:                                              ; preds = %592
  unreachable

594:                                              ; preds = %585
  %595 = add i64 %579, 40
  %596 = inttoptr i64 %587 to ptr
  br label %599

597:                                              ; preds = %592
  %598 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %598) ]
  unreachable

599:                                              ; preds = %594, %581
  %600 = phi i64 [ %595, %594 ], [ %583, %581 ]
  %601 = phi ptr [ %596, %594 ], [ %582, %581 ]
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %601, i64 noundef %600) #10
  br label %602

602:                                              ; preds = %577, %599
  store i64 0, ptr %235, align 8
  store i64 15, ptr %106, align 8
  store i8 0, ptr %11, align 8
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %11) #10
  call void @llvm.lifetime.end.p0(i64 2, ptr nonnull %9) #10
  ret void

603:                                              ; preds = %537, %497, %446, %400, %362, %314, %273, %227, %213, %149, %142
  %604 = cleanuppad within none []
  call void @"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %13) #10 [ "funclet"(token %604) ]
  cleanupret from %604 unwind label %605

605:                                              ; preds = %138, %105, %189, %603
  %606 = cleanuppad within none []
  call void @"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %11) #10 [ "funclet"(token %606) ]
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %11) #10
  cleanupret from %606 unwind to caller
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare dso_local double @frexp(double noundef, ptr noundef captures(none)) local_unnamed_addr #24

; Function Attrs: inlinehint mustprogress uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(32) ptr @"??$_Reallocate_grow_by@V<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@_KD@Z@_KD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??append@01@QEAAAEAV01@0D@Z@_KD@Z"(ptr noundef nonnull align 8 dereferenceable(32) %0, i64 noundef %1, i8 %2, i64 noundef %3, i8 noundef %4) local_unnamed_addr #15 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %7 = load i64, ptr %6, align 8
  %8 = sub i64 9223372036854775807, %7
  %9 = icmp ult i64 %8, %1
  br i1 %9, label %10, label %11

10:                                               ; preds = %5
  tail call void @"?_Xlen_string@std@@YAXXZ"() #29
  unreachable

11:                                               ; preds = %5
  %12 = add i64 %7, %1
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %14 = load i64, ptr %13, align 8
  %15 = or i64 %12, 15
  %16 = icmp slt i64 %12, 0
  br i1 %16, label %24, label %17

17:                                               ; preds = %11
  %18 = lshr i64 %14, 1
  %19 = xor i64 %18, 9223372036854775807
  %20 = icmp ugt i64 %14, %19
  br i1 %20, label %24, label %21

21:                                               ; preds = %17
  %22 = add i64 %18, %14
  %23 = tail call i64 @llvm.umax.i64(i64 %15, i64 %22)
  br label %24

24:                                               ; preds = %11, %17, %21
  %25 = phi i64 [ %23, %21 ], [ 9223372036854775807, %11 ], [ 9223372036854775807, %17 ]
  %26 = add i64 %25, 1
  %27 = icmp eq i64 %26, 0
  br i1 %27, label %43, label %28

28:                                               ; preds = %24
  %29 = icmp ugt i64 %26, 4095
  br i1 %29, label %30, label %41

30:                                               ; preds = %28
  %31 = icmp ult i64 %26, -39
  br i1 %31, label %33, label %32

32:                                               ; preds = %30
  tail call void @"?_Throw_bad_array_new_length@std@@YAXXZ"() #29
  unreachable

33:                                               ; preds = %30
  %34 = add i64 %25, 40
  %35 = tail call noalias noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef %34) #31
  %36 = ptrtoint ptr %35 to i64
  %37 = add i64 %36, 39
  %38 = and i64 %37, -32
  %39 = inttoptr i64 %38 to ptr
  %40 = getelementptr inbounds i8, ptr %39, i64 -8
  store i64 %36, ptr %40, align 8
  br label %43

41:                                               ; preds = %28
  %42 = tail call noalias noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef %26) #31
  br label %43

43:                                               ; preds = %24, %33, %41
  %44 = phi ptr [ %39, %33 ], [ %42, %41 ], [ null, %24 ]
  store i64 %12, ptr %6, align 8
  store i64 %25, ptr %13, align 8
  %45 = icmp ugt i64 %14, 15
  br i1 %45, label %46, label %69

46:                                               ; preds = %43
  %47 = load ptr, ptr %0, align 8
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 1 %44, ptr align 1 %47, i64 %7, i1 false)
  %48 = getelementptr i8, ptr %44, i64 %7
  tail call void @llvm.memset.p0.i64(ptr align 1 %48, i8 %4, i64 %3, i1 false)
  %49 = getelementptr i8, ptr %48, i64 %3
  store i8 0, ptr %49, align 1
  %50 = add i64 %14, 1
  %51 = icmp ugt i64 %50, 4095
  br i1 %51, label %52, label %66

52:                                               ; preds = %46
  %53 = getelementptr inbounds i8, ptr %47, i64 -8
  %54 = load i64, ptr %53, align 8
  %55 = ptrtoint ptr %47 to i64
  %56 = add i64 %55, -8
  %57 = sub i64 %56, %54
  %58 = icmp ult i64 %57, 32
  br i1 %58, label %61, label %59

59:                                               ; preds = %52
  invoke void @_invalid_parameter_noinfo_noreturn() #29
          to label %60 unwind label %64

60:                                               ; preds = %59
  unreachable

61:                                               ; preds = %52
  %62 = add i64 %14, 40
  %63 = inttoptr i64 %54 to ptr
  br label %66

64:                                               ; preds = %59
  %65 = cleanuppad within none []
  call void @__std_terminate() #30 [ "funclet"(token %65) ]
  unreachable

66:                                               ; preds = %46, %61
  %67 = phi i64 [ %62, %61 ], [ %50, %46 ]
  %68 = phi ptr [ %63, %61 ], [ %47, %46 ]
  tail call void @"??3@YAXPEAX_K@Z"(ptr noundef %68, i64 noundef %67) #10
  br label %72

69:                                               ; preds = %43
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 1 %44, ptr nonnull align 8 %0, i64 %7, i1 false)
  %70 = getelementptr i8, ptr %44, i64 %7
  tail call void @llvm.memset.p0.i64(ptr align 1 %70, i8 %4, i64 %3, i1 false)
  %71 = getelementptr i8, ptr %70, i64 %3
  store i8 0, ptr %71, align 1
  br label %72

72:                                               ; preds = %69, %66
  store ptr %44, ptr %0, align 8
  ret ptr %0
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare dso_local i64 @strcspn(ptr noundef captures(none), ptr noundef captures(none)) local_unnamed_addr #8

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #25

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umax.i64(i64, i64) #25

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #26

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.usub.sat.i64(i64, i64) #25

attributes #0 = { mustprogress norecurse uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { nobuiltin allocsize(0) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { nobuiltin nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #7 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #8 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #9 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #10 = { nounwind }
attributes #11 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #12 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #13 = { inlinehint mustprogress noreturn uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #14 = { noreturn "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #15 = { inlinehint mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #16 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #17 = { mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #18 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #19 = { mustprogress nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #20 = { mustprogress noreturn uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #21 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #22 = { mustprogress noinline nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #23 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #24 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #25 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #26 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #27 = { builtin allocsize(0) }
attributes #28 = { builtin nounwind }
attributes #29 = { noreturn }
attributes #30 = { noreturn nounwind }
attributes #31 = { allocsize(0) }
attributes #32 = { allocsize(0,1) }

!llvm.dbg.cu = !{!0}
!llvm.linker.options = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.module.flags = !{!9, !10, !11, !12, !13}
!llvm.ident = !{!14}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 22.0.0git (https://github.com/llvm/llvm-project.git 5223317210cca7705d43fde4005270f5bb45215b)", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "PointerTest.cpp", directory: "D:\\CMakeAndLLVM\\llvm-project\\llvm\\test\\Transforms\\PointerTypeTests")
!2 = !{!"/FAILIFMISMATCH:\22_MSC_VER=1900\22"}
!3 = !{!"/FAILIFMISMATCH:\22_ITERATOR_DEBUG_LEVEL=0\22"}
!4 = !{!"/FAILIFMISMATCH:\22RuntimeLibrary=MT_StaticRelease\22"}
!5 = !{!"/DEFAULTLIB:libcpmt.lib"}
!6 = !{!"/FAILIFMISMATCH:\22_CRT_STDIO_ISO_WIDE_SPECIFIERS=0\22"}
!7 = !{!"/FAILIFMISMATCH:\22annotate_string=0\22"}
!8 = !{!"/FAILIFMISMATCH:\22annotate_vector=0\22"}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 2}
!11 = !{i32 8, !"PIC Level", i32 2}
!12 = !{i32 7, !"uwtable", i32 2}
!13 = !{i32 1, !"MaxTLSAlign", i32 65536}
!14 = !{!"clang version 22.0.0git (https://github.com/llvm/llvm-project.git 5223317210cca7705d43fde4005270f5bb45215b)"}
!15 = distinct !{!15, !16, !17}
!16 = !{!"llvm.loop.mustprogress"}
!17 = !{!"llvm.loop.unroll.disable"}
!18 = !{i8 0, i8 2}
!19 = !{}
!20 = !{i64 8}
!21 = !{!22}
!22 = distinct !{!22, !23, !"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ: argument 0"}
!23 = distinct !{!23, !"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"}
!24 = !{!25}
!25 = distinct !{!25, !26, !"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ: argument 0"}
!26 = distinct !{!26, !"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"}
!27 = !{!28}
!28 = distinct !{!28, !29, !"?put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DJ@Z: argument 0"}
!29 = distinct !{!29, !"?put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DJ@Z"}
!30 = !{!31}
!31 = distinct !{!31, !32, !"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ: argument 0"}
!32 = distinct !{!32, !"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"}
!33 = !{!34}
!34 = distinct !{!34, !35, !"?put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBX@Z: argument 0"}
!35 = distinct !{!35, !"?put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBX@Z"}
!36 = !{!"branch_weights", i32 1, i32 1048575}
!37 = !{!38}
!38 = distinct !{!38, !39, !"?message@error_code@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ: argument 0"}
!39 = distinct !{!39, !"?message@error_code@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"}
!40 = distinct !{!40, !16, !17}
!41 = distinct !{!41, !16, !17}
!42 = !{!43}
!43 = distinct !{!43, !44, !"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ: argument 0"}
!44 = distinct !{!44, !"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"}
!45 = !{!46}
!46 = distinct !{!46, !47, !"?truename@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ: argument 0"}
!47 = distinct !{!47, !"?truename@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"}
!48 = !{!49}
!49 = distinct !{!49, !50, !"?falsename@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ: argument 0"}
!50 = distinct !{!50, !"?falsename@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"}
!51 = !{!52}
!52 = distinct !{!52, !53, !"?_Rep@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@D_K@Z: argument 0"}
!53 = distinct !{!53, !"?_Rep@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@D_K@Z"}
!54 = distinct !{!54, !16, !17}
!55 = !{!56}
!56 = distinct !{!56, !57, !"?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z: argument 0"}
!57 = distinct !{!57, !"?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z"}
!58 = distinct !{!58, !16, !17}
!59 = !{!60}
!60 = distinct !{!60, !61, !"?_Rep@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@D_K@Z: argument 0"}
!61 = distinct !{!61, !"?_Rep@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@D_K@Z"}
!62 = !{!63}
!63 = distinct !{!63, !64, !"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ: argument 0"}
!64 = distinct !{!64, !"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"}
!65 = !{!66}
!66 = distinct !{!66, !67, !"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ: argument 0"}
!67 = distinct !{!67, !"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"}
!68 = !{!69}
!69 = distinct !{!69, !70, !"?grouping@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ: argument 0"}
!70 = distinct !{!70, !"?grouping@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"}
!71 = distinct !{!71, !16, !17}
!72 = !{!73}
!73 = distinct !{!73, !74, !"?_Rep@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@D_K@Z: argument 0"}
!74 = distinct !{!74, !"?_Rep@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@D_K@Z"}
!75 = !{!76}
!76 = distinct !{!76, !77, !"?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z: argument 0"}
!77 = distinct !{!77, !"?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z"}
!78 = !{!79}
!79 = distinct !{!79, !80, !"?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z: argument 0"}
!80 = distinct !{!80, !"?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z"}
!81 = !{!82}
!82 = distinct !{!82, !83, !"?_Rep@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@D_K@Z: argument 0"}
!83 = distinct !{!83, !"?_Rep@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@D_K@Z"}
!84 = !{!85}
!85 = distinct !{!85, !86, !"?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z: argument 0"}
!86 = distinct !{!86, !"?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z"}
!87 = !{!88}
!88 = distinct !{!88, !89, !"?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z: argument 0"}
!89 = distinct !{!89, !"?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z"}
!90 = !{!91}
!91 = distinct !{!91, !92, !"?_Rep@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@D_K@Z: argument 0"}
!92 = distinct !{!92, !"?_Rep@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@D_K@Z"}
!93 = !{!94}
!94 = distinct !{!94, !95, !"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ: argument 0"}
!95 = distinct !{!95, !"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"}
!96 = !{!97}
!97 = distinct !{!97, !98, !"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ: argument 0"}
!98 = distinct !{!98, !"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"}
!99 = !{!100}
!100 = distinct !{!100, !101, !"?grouping@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ: argument 0"}
!101 = distinct !{!101, !"?grouping@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"}
!102 = distinct !{!102, !16, !17}
!103 = !{!104}
!104 = distinct !{!104, !105, !"?_Rep@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@D_K@Z: argument 0"}
!105 = distinct !{!105, !"?_Rep@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@D_K@Z"}
!106 = !{!107}
!107 = distinct !{!107, !108, !"?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z: argument 0"}
!108 = distinct !{!108, !"?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z"}
!109 = !{!110}
!110 = distinct !{!110, !111, !"?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z: argument 0"}
!111 = distinct !{!111, !"?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z"}
!112 = !{!113}
!113 = distinct !{!113, !114, !"?_Rep@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@D_K@Z: argument 0"}
!114 = distinct !{!114, !"?_Rep@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@D_K@Z"}
!115 = !{!116}
!116 = distinct !{!116, !117, !"?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z: argument 0"}
!117 = distinct !{!117, !"?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z"}
!118 = !{!119}
!119 = distinct !{!119, !120, !"?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z: argument 0"}
!120 = distinct !{!120, !"?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z"}
!121 = !{!122}
!122 = distinct !{!122, !123, !"?_Rep@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@D_K@Z: argument 0"}
!123 = distinct !{!123, !"?_Rep@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@D_K@Z"}
