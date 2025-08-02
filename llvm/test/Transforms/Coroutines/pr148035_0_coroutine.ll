; This is the cppreference example for coroutines, in llvm IR form, built with async exceptions flag.
; crashed before fix because of the validation mismatch of Unwind edges out of a funclet pad must have the same unwind dest
; RUN: opt < %s -passes=coro-split -S | FileCheck %s
; CHECK: define

; ModuleID = 'coroutine.cpp'
source_filename = "coroutine.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.38.33135"

%"class.std::basic_ostream" = type { ptr, [4 x i8], i32, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", ptr, ptr, i8 }
%"class.std::ios_base" = type { ptr, i64, i32, i32, i32, i64, i64, ptr, ptr, ptr }
%rtti.TypeDescriptor23 = type { ptr, ptr, [24 x i8] }
%eh.CatchableType = type { i32, i32, i32, i32, i32, i32, i32 }
%rtti.TypeDescriptor19 = type { ptr, ptr, [20 x i8] }
%eh.CatchableTypeArray.2 = type { i32, [2 x i32] }
%eh.ThrowInfo = type { i32, i32, i32, i32 }
%rtti.CompleteObjectLocator = type { i32, i32, i32, i32, i32, i32 }
%rtti.ClassHierarchyDescriptor = type { i32, i32, i32, i32 }
%rtti.BaseClassDescriptor = type { i32, i32, i32, i32, i32, i32, i32 }
%"struct.std::nostopstate_t" = type { i8 }
%rtti.TypeDescriptor26 = type { ptr, ptr, [27 x i8] }
%rtti.TypeDescriptor22 = type { ptr, ptr, [23 x i8] }
%eh.CatchableTypeArray.5 = type { i32, [5 x i32] }
%"union.std::error_category::_Addr_storage" = type { i64 }
%rtti.TypeDescriptor35 = type { ptr, ptr, [36 x i8] }
%rtti.TypeDescriptor24 = type { ptr, ptr, [25 x i8] }
%"struct.std::_Fake_allocator" = type { i8 }
%rtti.TypeDescriptor30 = type { ptr, ptr, [31 x i8] }
%eh.CatchableTypeArray.3 = type { i32, [3 x i32] }
%struct.awaitable = type { ptr }
%struct.task = type { i8 }
%"struct.task::promise_type" = type { i8 }
%"struct.std::suspend_never" = type { i8 }
%"class.std::thread::id" = type { i32 }
%"struct.std::coroutine_handle" = type { ptr }
%"struct.std::coroutine_handle.0" = type { ptr }
%"class.std::basic_ostream<char>::sentry" = type { %"class.std::basic_ostream<char>::_Sentry_base", i8 }
%"class.std::basic_ostream<char>::_Sentry_base" = type { ptr }
%"class.std::runtime_error" = type { %"class.std::exception" }
%"class.std::exception" = type { ptr, %struct.__std_exception_data }
%struct.__std_exception_data = type { ptr, i8 }
%"class.std::jthread" = type { %"class.std::thread", %"class.std::stop_source" }
%"class.std::thread" = type { %struct._Thrd_t }
%struct._Thrd_t = type { ptr, i32 }
%"class.std::stop_source" = type { ptr }
%class.anon = type { %"struct.std::coroutine_handle" }
%"class.std::unique_ptr" = type { %"class.std::_Compressed_pair" }
%"class.std::_Compressed_pair" = type { ptr }
%"struct.std::_Stop_state" = type { %"struct.std::atomic", %"struct.std::atomic", %"class.std::_Locked_pointer", %"struct.std::atomic.6", i32 }
%"struct.std::atomic" = type { %"struct.std::_Atomic_integral_facade" }
%"struct.std::_Atomic_integral_facade" = type { %"struct.std::_Atomic_integral" }
%"struct.std::_Atomic_integral" = type { %"struct.std::_Atomic_storage" }
%"struct.std::_Atomic_storage" = type { %"struct.std::_Atomic_padded" }
%"struct.std::_Atomic_padded" = type { i32 }
%"class.std::_Locked_pointer" = type { %"struct.std::atomic.1" }
%"struct.std::atomic.1" = type { %"struct.std::_Atomic_integral_facade.2" }
%"struct.std::_Atomic_integral_facade.2" = type { %"struct.std::_Atomic_integral.3" }
%"struct.std::_Atomic_integral.3" = type { %"struct.std::_Atomic_storage.4" }
%"struct.std::_Atomic_storage.4" = type { %"struct.std::_Atomic_padded.5" }
%"struct.std::_Atomic_padded.5" = type { i64 }
%"struct.std::atomic.6" = type { %"struct.std::_Atomic_pointer" }
%"struct.std::_Atomic_pointer" = type { %"struct.std::_Atomic_storage.7" }
%"struct.std::_Atomic_storage.7" = type { %"struct.std::_Atomic_padded.8" }
%"struct.std::_Atomic_padded.8" = type { ptr }
%"struct.std::_Exact_args_t" = type { i8 }
%"struct.std::_Zero_then_variadic_args_t" = type { i8 }
%"class.std::tuple" = type { %"struct.std::_Tuple_val" }
%"struct.std::_Tuple_val" = type { %class.anon }
%"class.std::_Stop_callback_base" = type { ptr, ptr, ptr, ptr }
%"class.std::basic_streambuf" = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr }
%"class.std::ios_base::failure" = type { %"class.std::system_error" }
%"class.std::system_error" = type { %"class.std::_System_error" }
%"class.std::_System_error" = type { %"class.std::runtime_error", %"class.std::error_code" }
%"class.std::error_code" = type { i32, ptr }
%"class.std::basic_string" = type { %"class.std::_Compressed_pair.10" }
%"class.std::_Compressed_pair.10" = type { %"class.std::_String_val" }
%"class.std::_String_val" = type { %"union.std::_String_val<std::_Simple_types<char>>::_Bxty", i64, i64 }
%"union.std::_String_val<std::_Simple_types<char>>::_Bxty" = type { ptr, [8 x i8] }
%"class.std::error_condition" = type { i32, ptr }
%"struct.std::_Fake_proxy_ptr_impl" = type { i8 }
%"class.std::bad_array_new_length" = type { %"class.std::bad_alloc" }
%"class.std::bad_alloc" = type { %"class.std::exception" }
%"class.std::error_category" = type { ptr, %"union.std::error_category::_Addr_storage" }
%"class.std::allocator" = type { i8 }
%"struct.std::_One_then_variadic_args_t" = type { i8 }
%class.anon.11 = type { i8 }

$"?get_return_object@promise_type@task@@QEAA?AU2@XZ" = comdat any

$"?initial_suspend@promise_type@task@@QEAA?AUsuspend_never@std@@XZ" = comdat any

$"?await_ready@suspend_never@std@@QEBA_NXZ" = comdat any

$"?await_suspend@suspend_never@std@@QEBAXU?$coroutine_handle@X@2@@Z" = comdat any

$"?from_address@?$coroutine_handle@Upromise_type@task@@@std@@SA?AU12@QEAX@Z" = comdat any

$"??B?$coroutine_handle@Upromise_type@task@@@std@@QEBA?AU?$coroutine_handle@X@1@XZ" = comdat any

$"?await_resume@suspend_never@std@@QEBAXXZ" = comdat any

$"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@D@Z" = comdat any

$"??$?6DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@Vid@thread@0@@Z" = comdat any

$"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z" = comdat any

$"?get_id@this_thread@std@@YA?AVid@thread@2@XZ" = comdat any

$"?return_void@promise_type@task@@QEAAXXZ" = comdat any

$"?unhandled_exception@promise_type@task@@QEAAXXZ" = comdat any

$"?final_suspend@promise_type@task@@QEAA?AUsuspend_never@std@@XZ" = comdat any

$"??0jthread@std@@QEAA@XZ" = comdat any

$"??1jthread@std@@QEAA@XZ" = comdat any

$"??0?$coroutine_handle@Upromise_type@task@@@std@@QEAA@XZ" = comdat any

$"?from_address@?$coroutine_handle@X@std@@SA?AU12@QEAX@Z" = comdat any

$"??0?$coroutine_handle@X@std@@QEAA@XZ" = comdat any

$"??0id@thread@std@@AEAA@I@Z" = comdat any

$"?joinable@jthread@std@@QEBA_NXZ" = comdat any

$"??0runtime_error@std@@QEAA@PEBD@Z" = comdat any

$"??0runtime_error@std@@QEAA@AEBV01@@Z" = comdat any

$"??0exception@std@@QEAA@AEBV01@@Z" = comdat any

$"??1runtime_error@std@@UEAA@XZ" = comdat any

$"??4jthread@std@@QEAAAEAV01@$$QEAV01@@Z" = comdat any

$"?get_id@jthread@std@@QEBA?AVid@thread@2@XZ" = comdat any

$"?joinable@thread@std@@QEBA_NXZ" = comdat any

$"??0exception@std@@QEAA@QEBD@Z" = comdat any

$"??1exception@std@@UEAA@XZ" = comdat any

$"??_Gruntime_error@std@@UEAAPEAXI@Z" = comdat any

$"?what@exception@std@@UEBAPEBDXZ" = comdat any

$"??_Gexception@std@@UEAAPEAXI@Z" = comdat any

$"??0thread@std@@QEAA@XZ" = comdat any

$"??0stop_source@std@@QEAA@XZ" = comdat any

$"??1stop_source@std@@QEAA@XZ" = comdat any

$"??1thread@std@@QEAA@XZ" = comdat any

$"??0_Stop_state@std@@QEAA@XZ" = comdat any

$"??0?$_Locked_pointer@V_Stop_callback_base@std@@@std@@QEAA@XZ" = comdat any

$"??0?$_Atomic_storage@I$03@std@@QEAA@I@Z" = comdat any

$"??0?$atomic@_K@std@@QEAA@XZ" = comdat any

$"??0?$_Atomic_storage@PEBV_Stop_callback_base@std@@$07@std@@QEAA@QEBV_Stop_callback_base@1@@Z" = comdat any

$"??$?0U_Exact_args_t@std@@$0A@@?$tuple@$$V@std@@QEAA@U_Exact_args_t@1@@Z" = comdat any

$"?resume@?$coroutine_handle@X@std@@QEBAXXZ" = comdat any

$"?fetch_sub@?$_Atomic_integral_facade@I@std@@QEAAIIW4memory_order@2@@Z" = comdat any

$"?fetch_add@?$_Atomic_integral@I$03@std@@QEAAIIW4memory_order@2@@Z" = comdat any

$"?_Negate@?$_Atomic_integral_facade@I@std@@SAII@Z" = comdat any

$_Check_memory_order = comdat any

$"??$_Atomic_address_as@JU?$_Atomic_padded@I@std@@@std@@YAPECJAEAU?$_Atomic_padded@I@0@@Z" = comdat any

$"?_Try_cancel_and_join@jthread@std@@AEAAXXZ" = comdat any

$"??4thread@std@@QEAAAEAV01@$$QEAV01@@Z" = comdat any

$"??4stop_source@std@@QEAAAEAV01@$$QEAV01@@Z" = comdat any

$"?request_stop@stop_source@std@@QEAA_NXZ" = comdat any

$"?join@thread@std@@QEAAXXZ" = comdat any

$"?_Request_stop@_Stop_state@std@@QEAA_NXZ" = comdat any

$"?fetch_or@?$_Atomic_integral@I$03@std@@QEAAIIW4memory_order@2@@Z" = comdat any

$"?_Lock_and_load@?$_Locked_pointer@V_Stop_callback_base@std@@@std@@QEAAPEAV_Stop_callback_base@2@XZ" = comdat any

$"?store@?$_Atomic_storage@PEBV_Stop_callback_base@std@@$07@std@@QEAAXQEBV_Stop_callback_base@2@W4memory_order@2@@Z" = comdat any

$"?notify_all@?$_Atomic_storage@PEBV_Stop_callback_base@std@@$07@std@@QEAAXXZ" = comdat any

$"?_Store_and_unlock@?$_Locked_pointer@V_Stop_callback_base@std@@@std@@QEAAXQEAV_Stop_callback_base@2@@Z" = comdat any

$"??$exchange@PEAV_Stop_callback_base@std@@$$T@std@@YAPEAV_Stop_callback_base@0@AEAPEAV10@$$QEA$$T@Z" = comdat any

$"?load@?$_Atomic_storage@_K$07@std@@QEBA_KW4memory_order@2@@Z" = comdat any

$"?compare_exchange_weak@?$atomic@_K@std@@QEAA_NAEA_K_K@Z" = comdat any

$"?wait@?$_Atomic_storage@_K$07@std@@QEBAX_KW4memory_order@2@@Z" = comdat any

$"??$_Atomic_address_as@_JU?$_Atomic_padded@_K@std@@@std@@YAPED_JAEBU?$_Atomic_padded@_K@0@@Z" = comdat any

$"?compare_exchange_strong@?$_Atomic_storage@_K$07@std@@QEAA_NAEA_K_KW4memory_order@2@@Z" = comdat any

$"??$_Atomic_reinterpret_as@_J_K@std@@YA_JAEB_K@Z" = comdat any

$"??$_Atomic_address_as@_JU?$_Atomic_padded@_K@std@@@std@@YAPEC_JAEAU?$_Atomic_padded@_K@0@@Z" = comdat any

$"??$_Atomic_wait_direct@_K_J@std@@YAXQEBU?$_Atomic_storage@_K$07@0@_JW4memory_order@0@@Z" = comdat any

$"??$_Atomic_address_as@_JU?$_Atomic_padded@PEBV_Stop_callback_base@std@@@std@@@std@@YAPEC_JAEAU?$_Atomic_padded@PEBV_Stop_callback_base@std@@@0@@Z" = comdat any

$"??$_Atomic_reinterpret_as@_JPEBV_Stop_callback_base@std@@@std@@YA_JAEBQEBV_Stop_callback_base@0@@Z" = comdat any

$"?store@?$_Atomic_storage@PEBV_Stop_callback_base@std@@$07@std@@QEAAXQEBV_Stop_callback_base@2@@Z" = comdat any

$"?exchange@?$_Atomic_storage@_K$07@std@@QEAA_K_KW4memory_order@2@@Z" = comdat any

$"?notify_all@?$_Atomic_storage@_K$07@std@@QEAAXXZ" = comdat any

$"??$exchange@U_Thrd_t@@U1@@std@@YA?AU_Thrd_t@@AEAU1@$$QEAU1@@Z" = comdat any

$"??0stop_source@std@@QEAA@$$QEAV01@@Z" = comdat any

$"?swap@stop_source@std@@QEAAXAEAV12@@Z" = comdat any

$"??$exchange@PEAU_Stop_state@std@@$$T@std@@YAPEAU_Stop_state@0@AEAPEAU10@$$QEA$$T@Z" = comdat any

$"??$swap@PEAU_Stop_state@std@@$0A@@std@@YAXAEAPEAU_Stop_state@0@0@Z" = comdat any

$"?get_id@thread@std@@QEBA?AVid@12@XZ" = comdat any

$"??0stop_source@std@@QEAA@Unostopstate_t@1@@Z" = comdat any

$"?length@?$_Narrow_char_traits@DH@std@@SA_KQEBD@Z" = comdat any

$"?width@ios_base@std@@QEBA_JXZ" = comdat any

$"??0sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z" = comdat any

$"??Bsentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEBA_NXZ" = comdat any

$"?flags@ios_base@std@@QEBAHXZ" = comdat any

$"?eq_int_type@?$_Narrow_char_traits@DH@std@@SA_NHH@Z" = comdat any

$"?rdbuf@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_streambuf@DU?$char_traits@D@std@@@2@XZ" = comdat any

$"?sputc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHD@Z" = comdat any

$"?fill@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADXZ" = comdat any

$"?eof@?$_Narrow_char_traits@DH@std@@SAHXZ" = comdat any

$"?sputn@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAA_JPEBD_J@Z" = comdat any

$"?width@ios_base@std@@QEAA_J_J@Z" = comdat any

$"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z" = comdat any

$"??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ" = comdat any

$"??0_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z" = comdat any

$"?good@ios_base@std@@QEBA_NXZ" = comdat any

$"?tie@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_ostream@DU?$char_traits@D@std@@@2@XZ" = comdat any

$"?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ" = comdat any

$"??1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ" = comdat any

$"?rdstate@ios_base@std@@QEBAHXZ" = comdat any

$"?pubsync@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ" = comdat any

$"?_Pnavail@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEBA_JXZ" = comdat any

$"?to_int_type@?$_Narrow_char_traits@DH@std@@SAHD@Z" = comdat any

$"?_Pninc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAPEADXZ" = comdat any

$"?clear@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z" = comdat any

$"?clear@ios_base@std@@QEAAXH_N@Z" = comdat any

$"?make_error_code@std@@YA?AVerror_code@1@W4io_errc@1@@Z" = comdat any

$"??0failure@ios_base@std@@QEAA@PEBDAEBVerror_code@2@@Z" = comdat any

$"??0failure@ios_base@std@@QEAA@AEBV012@@Z" = comdat any

$"??0system_error@std@@QEAA@AEBV01@@Z" = comdat any

$"??0_System_error@std@@QEAA@AEBV01@@Z" = comdat any

$"??1failure@ios_base@std@@UEAA@XZ" = comdat any

$"?iostream_category@std@@YAAEBVerror_category@1@XZ" = comdat any

$"??0error_code@std@@QEAA@HAEBVerror_category@1@@Z" = comdat any

$"??$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@0@XZ" = comdat any

$"??_G_Iostream_error_category2@std@@UEAAPEAXI@Z" = comdat any

$"?name@_Iostream_error_category2@std@@UEBAPEBDXZ" = comdat any

$"?message@_Iostream_error_category2@std@@UEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@H@Z" = comdat any

$"?default_error_condition@error_category@std@@UEBA?AVerror_condition@2@H@Z" = comdat any

$"?equivalent@error_category@std@@UEBA_NAEBVerror_code@2@H@Z" = comdat any

$"?equivalent@error_category@std@@UEBA_NHAEBVerror_condition@2@@Z" = comdat any

$"??1_Iostream_error_category2@std@@UEAA@XZ" = comdat any

$"??1error_category@std@@UEAA@XZ" = comdat any

$"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD_K@Z" = comdat any

$"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD@Z" = comdat any

$"??$?0$$V@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@U_Zero_then_variadic_args_t@1@@Z" = comdat any

$"??$_Construct@$00PEBD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXQEBD_K@Z" = comdat any

$"??1?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@XZ" = comdat any

$"??0?$allocator@D@std@@QEAA@XZ" = comdat any

$"??0?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ" = comdat any

$"??1?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ" = comdat any

$"??0_Bxty@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ" = comdat any

$"??1_Bxty@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ" = comdat any

$"?max_size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ" = comdat any

$"?_Xlen_string@std@@YAXXZ" = comdat any

$"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV?$allocator@D@2@XZ" = comdat any

$"??0_Fake_proxy_ptr_impl@std@@QEAA@AEBU_Fake_allocator@1@AEBU_Container_base0@1@@Z" = comdat any

$"?copy@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z" = comdat any

$"?assign@?$_Narrow_char_traits@DH@std@@SAXAEADAEBD@Z" = comdat any

$"?_Release@_Fake_proxy_ptr_impl@std@@QEAAXXZ" = comdat any

$"?_Calculate_growth@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBA_K_K@Z" = comdat any

$"??$_Allocate_for_capacity@$0A@@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@CAPEADAEAV?$allocator@D@1@AEA_K@Z" = comdat any

$"??$_Construct_in_place@PEADAEBQEAD@std@@YAXAEAPEADAEBQEAD@Z" = comdat any

$"??$_Unfancy@D@std@@YAPEADPEAD@Z" = comdat any

$"?max_size@?$_Default_allocator_traits@V?$allocator@D@std@@@std@@SA_KAEBV?$allocator@D@2@@Z" = comdat any

$"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBAAEBV?$allocator@D@2@XZ" = comdat any

$"??$max@_K@std@@YAAEB_KAEB_K0@Z" = comdat any

$"??$min@_K@std@@YAAEB_KAEB_K0@Z" = comdat any

$"?max@?$numeric_limits@_J@std@@SA_JXZ" = comdat any

$"?_Get_first@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEBAAEBV?$allocator@D@2@XZ" = comdat any

$"?_Get_first@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAAAEAV?$allocator@D@2@XZ" = comdat any

$"?_Calculate_growth@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@CA_K_K00@Z" = comdat any

$"??$_Allocate_at_least_helper@V?$allocator@D@std@@@std@@YAPEADAEAV?$allocator@D@0@AEA_K@Z" = comdat any

$"?allocate@?$allocator@D@std@@QEAAPEAD_K@Z" = comdat any

$"??$_Allocate@$0BA@U_Default_allocate_traits@std@@$0A@@std@@YAPEAX_K@Z" = comdat any

$"??$_Get_size_of_n@$00@std@@YA_K_K@Z" = comdat any

$"??$_Allocate_manually_vector_aligned@U_Default_allocate_traits@std@@@std@@YAPEAX_K@Z" = comdat any

$"?_Allocate@_Default_allocate_traits@std@@SAPEAX_K@Z" = comdat any

$"?_Throw_bad_array_new_length@std@@YAXXZ" = comdat any

$"??0bad_array_new_length@std@@QEAA@XZ" = comdat any

$"??0bad_array_new_length@std@@QEAA@AEBV01@@Z" = comdat any

$"??0bad_alloc@std@@QEAA@AEBV01@@Z" = comdat any

$"??1bad_array_new_length@std@@UEAA@XZ" = comdat any

$"??0bad_alloc@std@@AEAA@QEBD@Z" = comdat any

$"??1bad_alloc@std@@UEAA@XZ" = comdat any

$"??_Gbad_array_new_length@std@@UEAAPEAXI@Z" = comdat any

$"??0exception@std@@QEAA@QEBDH@Z" = comdat any

$"??_Gbad_alloc@std@@UEAAPEAXI@Z" = comdat any

$"??$_Convert_size@_K_K@std@@YA_K_K@Z" = comdat any

$"??0error_condition@std@@QEAA@HAEBVerror_category@1@@Z" = comdat any

$"??8error_category@std@@QEBA_NAEBV01@@Z" = comdat any

$"?category@error_code@std@@QEBAAEBVerror_category@2@XZ" = comdat any

$"?value@error_code@std@@QEBAHXZ" = comdat any

$"??$_Bit_cast@_KT_Addr_storage@error_category@std@@$0A@@std@@YA_KAEBT_Addr_storage@error_category@0@@Z" = comdat any

$"??8std@@YA_NAEBVerror_condition@0@0@Z" = comdat any

$"?category@error_condition@std@@QEBAAEBVerror_category@2@XZ" = comdat any

$"?value@error_condition@std@@QEBAHXZ" = comdat any

$"??0system_error@std@@QEAA@Verror_code@1@PEBD@Z" = comdat any

$"??1system_error@std@@UEAA@XZ" = comdat any

$"??_Gfailure@ios_base@std@@UEAAPEAXI@Z" = comdat any

$"??0_System_error@std@@IEAA@Verror_code@1@AEBV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@1@@Z" = comdat any

$"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ" = comdat any

$"??1_System_error@std@@UEAA@XZ" = comdat any

$"??_Gsystem_error@std@@UEAAPEAXI@Z" = comdat any

$"?_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z" = comdat any

$"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@AEBV01@@Z" = comdat any

$"??0runtime_error@std@@QEAA@AEBV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@1@@Z" = comdat any

$"??_G_System_error@std@@UEAAPEAXI@Z" = comdat any

$"?empty@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_NXZ" = comdat any

$"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@QEBD@Z" = comdat any

$"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@AEBV12@@Z" = comdat any

$"?message@error_code@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ" = comdat any

$"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@$$QEAV01@@Z" = comdat any

$"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@QEBD_K@Z" = comdat any

$"?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAAPEADXZ" = comdat any

$"?move@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z" = comdat any

$"??$_Reallocate_grow_by@V<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@QEBD_K@Z@PEBD_K@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??append@01@QEAAAEAV01@QEBD0@Z@PEBD_K@Z" = comdat any

$"?_Large_mode_engaged@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBA_NXZ" = comdat any

$"?_Orphan_all@_Container_base0@std@@QEAAXXZ" = comdat any

$"??R<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@QEBD_K@Z@QEBA?A?<auto>@@QEAD0101@Z" = comdat any

$"?_Deallocate_for_capacity@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@CAXAEAV?$allocator@D@2@QEAD_K@Z" = comdat any

$"?deallocate@?$allocator@D@std@@QEAAXQEAD_K@Z" = comdat any

$"??$_Deallocate@$0BA@$0A@@std@@YAXPEAX_K@Z" = comdat any

$"?_Adjust_manually_vector_aligned@std@@YAXAEAPEAXAEA_K@Z" = comdat any

$"?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBAPEBDXZ" = comdat any

$"??$?0V?$allocator@D@std@@$$V@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@U_One_then_variadic_args_t@1@$$QEAV?$allocator@D@1@@Z" = comdat any

$"?_Alloc_proxy@_Container_base0@std@@QEAAXAEBU_Fake_allocator@2@@Z" = comdat any

$"?_Take_contents@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXAEAV12@@Z" = comdat any

$"?_Memcpy_val_from@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXAEBV12@@Z" = comdat any

$"?_Tidy_init@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXXZ" = comdat any

$"?_Activate_SSO_buffer@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAAXXZ" = comdat any

$"?select_on_container_copy_construction@?$_Default_allocator_traits@V?$allocator@D@std@@@std@@SA?AV?$allocator@D@2@AEBV32@@Z" = comdat any

$"??$_Construct@$01PEBD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXQEBD_K@Z" = comdat any

$"?c_str@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBAPEBDXZ" = comdat any

$"?_Tidy_deallocate@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXXZ" = comdat any

$"??$_Destroy_in_place@PEAD@std@@YAXAEAPEAD@Z" = comdat any

$"?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ" = comdat any

$"??$end@D$0L@@std@@YAPEADAEAY0L@D@Z" = comdat any

$"??$_UIntegral_to_buff@DI@std@@YAPEADPEADI@Z" = comdat any

$"??_C@_0BO@EKOPNEHI@Coroutine?5started?5on?5thread?3?5?$AA@" = comdat any

$"??_C@_0BO@NLJBCPPM@Coroutine?5resumed?5on?5thread?3?5?$AA@" = comdat any

$"??_C@_0CD@HNLLMDJL@Output?5jthread?5parameter?5not?5emp@" = comdat any

$"??_R0?AVruntime_error@std@@@8" = comdat any

$"_CT??_R0?AVruntime_error@std@@@8??0runtime_error@std@@QEAA@AEBV01@@Z24" = comdat any

$"??_R0?AVexception@std@@@8" = comdat any

$"_CT??_R0?AVexception@std@@@8??0exception@std@@QEAA@AEBV01@@Z24" = comdat any

$"_CTA2?AVruntime_error@std@@" = comdat any

$"_TI2?AVruntime_error@std@@" = comdat any

$"??_C@_0BA@OADHDNAI@New?5thread?5ID?3?5?$AA@" = comdat any

$"??_7runtime_error@std@@6B@" = comdat largest

$"??_R4runtime_error@std@@6B@" = comdat any

$"??_R3runtime_error@std@@8" = comdat any

$"??_R2runtime_error@std@@8" = comdat any

$"??_R1A@?0A@EA@runtime_error@std@@8" = comdat any

$"??_R1A@?0A@EA@exception@std@@8" = comdat any

$"??_R3exception@std@@8" = comdat any

$"??_R2exception@std@@8" = comdat any

$"??_7exception@std@@6B@" = comdat largest

$"??_R4exception@std@@6B@" = comdat any

$"??_C@_0BC@EOODALEL@Unknown?5exception?$AA@" = comdat any

$"?nostopstate@std@@3Unostopstate_t@1@B" = comdat any

$"??_C@_0BF@PHHKMMFD@ios_base?3?3badbit?5set?$AA@" = comdat any

$"??_C@_0BG@FMKFHCIL@ios_base?3?3failbit?5set?$AA@" = comdat any

$"??_C@_0BF@OOHOMBOF@ios_base?3?3eofbit?5set?$AA@" = comdat any

$"??_R0?AVfailure@ios_base@std@@@8" = comdat any

$"_CT??_R0?AVfailure@ios_base@std@@@8??0failure@ios_base@std@@QEAA@AEBV012@@Z40" = comdat any

$"??_R0?AVsystem_error@std@@@8" = comdat any

$"_CT??_R0?AVsystem_error@std@@@8??0system_error@std@@QEAA@AEBV01@@Z40" = comdat any

$"??_R0?AV_System_error@std@@@8" = comdat any

$"_CT??_R0?AV_System_error@std@@@8??0_System_error@std@@QEAA@AEBV01@@Z40" = comdat any

$"_CTA5?AVfailure@ios_base@std@@" = comdat any

$"_TI5?AVfailure@ios_base@std@@" = comdat any

$"?_Static@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@1@XZ@4V21@B" = comdat any

$"??_7_Iostream_error_category2@std@@6B@" = comdat largest

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

$"?_Fake_alloc@std@@3U_Fake_allocator@1@B" = comdat any

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

$"??_7bad_alloc@std@@6B@" = comdat largest

$"??_R4bad_alloc@std@@6B@" = comdat any

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

$"??_7system_error@std@@6B@" = comdat largest

$"??_R4system_error@std@@6B@" = comdat any

$"??_7_System_error@std@@6B@" = comdat largest

$"??_R4_System_error@std@@6B@" = comdat any

$"??_C@_02LMMGGCAJ@?3?5?$AA@" = comdat any

@"?cout@std@@3V?$basic_ostream@DU?$char_traits@D@std@@@1@A" = external dso_local global %"class.std::basic_ostream", align 8
@"??_C@_0BO@EKOPNEHI@Coroutine?5started?5on?5thread?3?5?$AA@" = linkonce_odr dso_local unnamed_addr constant [30 x i8] c"Coroutine started on thread: \00", comdat, align 1
@"??_C@_0BO@NLJBCPPM@Coroutine?5resumed?5on?5thread?3?5?$AA@" = linkonce_odr dso_local unnamed_addr constant [30 x i8] c"Coroutine resumed on thread: \00", comdat, align 1
@"??_C@_0CD@HNLLMDJL@Output?5jthread?5parameter?5not?5emp@" = linkonce_odr dso_local unnamed_addr constant [35 x i8] c"Output jthread parameter not empty\00", comdat, align 1
@"??_7type_info@@6B@" = external constant ptr
@"??_R0?AVruntime_error@std@@@8" = linkonce_odr global %rtti.TypeDescriptor23 { ptr @"??_7type_info@@6B@", ptr null, [24 x i8] c".?AVruntime_error@std@@\00" }, comdat
@__ImageBase = external dso_local constant i8
@"_CT??_R0?AVruntime_error@std@@@8??0runtime_error@std@@QEAA@AEBV01@@Z24" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVruntime_error@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 -1, i32 0, i32 24, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??0runtime_error@std@@QEAA@AEBV01@@Z" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, section ".xdata", comdat
@"??_R0?AVexception@std@@@8" = linkonce_odr global %rtti.TypeDescriptor19 { ptr @"??_7type_info@@6B@", ptr null, [20 x i8] c".?AVexception@std@@\00" }, comdat
@"_CT??_R0?AVexception@std@@@8??0exception@std@@QEAA@AEBV01@@Z24" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVexception@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 -1, i32 0, i32 24, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??0exception@std@@QEAA@AEBV01@@Z" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, section ".xdata", comdat
@"_CTA2?AVruntime_error@std@@" = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.2 { i32 2, [2 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"_CT??_R0?AVruntime_error@std@@@8??0runtime_error@std@@QEAA@AEBV01@@Z24" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"_CT??_R0?AVexception@std@@@8??0exception@std@@QEAA@AEBV01@@Z24" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32)] }, section ".xdata", comdat
@"_TI2?AVruntime_error@std@@" = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??1runtime_error@std@@UEAA@XZ" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"_CTA2?AVruntime_error@std@@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, section ".xdata", comdat
@"??_C@_0BA@OADHDNAI@New?5thread?5ID?3?5?$AA@" = linkonce_odr dso_local unnamed_addr constant [16 x i8] c"New thread ID: \00", comdat, align 1
@0 = private unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr @"??_R4runtime_error@std@@6B@", ptr @"??_Gruntime_error@std@@UEAAPEAXI@Z", ptr @"?what@exception@std@@UEBAPEBDXZ"] }, comdat($"??_7runtime_error@std@@6B@")
@"??_R4runtime_error@std@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVruntime_error@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3runtime_error@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R4runtime_error@std@@6B@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R3runtime_error@std@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 0, i32 2, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R2runtime_error@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R2runtime_error@std@@8" = linkonce_odr constant [3 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@runtime_error@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@exception@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0], comdat
@"??_R1A@?0A@EA@runtime_error@std@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVruntime_error@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 1, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3runtime_error@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R1A@?0A@EA@exception@std@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVexception@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3exception@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R3exception@std@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 0, i32 1, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R2exception@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R2exception@std@@8" = linkonce_odr constant [2 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@exception@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0], comdat
@1 = private unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr @"??_R4exception@std@@6B@", ptr @"??_Gexception@std@@UEAAPEAXI@Z", ptr @"?what@exception@std@@UEBAPEBDXZ"] }, comdat($"??_7exception@std@@6B@")
@"??_R4exception@std@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVexception@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3exception@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R4exception@std@@6B@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_C@_0BC@EOODALEL@Unknown?5exception?$AA@" = linkonce_odr dso_local unnamed_addr constant [18 x i8] c"Unknown exception\00", comdat, align 1
@"?nostopstate@std@@3Unostopstate_t@1@B" = linkonce_odr dso_local constant %"struct.std::nostopstate_t" zeroinitializer, comdat, align 1
@"??_C@_0BF@PHHKMMFD@ios_base?3?3badbit?5set?$AA@" = linkonce_odr dso_local unnamed_addr constant [21 x i8] c"ios_base::badbit set\00", comdat, align 1
@"??_C@_0BG@FMKFHCIL@ios_base?3?3failbit?5set?$AA@" = linkonce_odr dso_local unnamed_addr constant [22 x i8] c"ios_base::failbit set\00", comdat, align 1
@"??_C@_0BF@OOHOMBOF@ios_base?3?3eofbit?5set?$AA@" = linkonce_odr dso_local unnamed_addr constant [21 x i8] c"ios_base::eofbit set\00", comdat, align 1
@"??_R0?AVfailure@ios_base@std@@@8" = linkonce_odr global %rtti.TypeDescriptor26 { ptr @"??_7type_info@@6B@", ptr null, [27 x i8] c".?AVfailure@ios_base@std@@\00" }, comdat
@"_CT??_R0?AVfailure@ios_base@std@@@8??0failure@ios_base@std@@QEAA@AEBV012@@Z40" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVfailure@ios_base@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 -1, i32 0, i32 40, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??0failure@ios_base@std@@QEAA@AEBV012@@Z" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, section ".xdata", comdat
@"??_R0?AVsystem_error@std@@@8" = linkonce_odr global %rtti.TypeDescriptor22 { ptr @"??_7type_info@@6B@", ptr null, [23 x i8] c".?AVsystem_error@std@@\00" }, comdat
@"_CT??_R0?AVsystem_error@std@@@8??0system_error@std@@QEAA@AEBV01@@Z40" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVsystem_error@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 -1, i32 0, i32 40, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??0system_error@std@@QEAA@AEBV01@@Z" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, section ".xdata", comdat
@"??_R0?AV_System_error@std@@@8" = linkonce_odr global %rtti.TypeDescriptor23 { ptr @"??_7type_info@@6B@", ptr null, [24 x i8] c".?AV_System_error@std@@\00" }, comdat
@"_CT??_R0?AV_System_error@std@@@8??0_System_error@std@@QEAA@AEBV01@@Z40" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AV_System_error@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 -1, i32 0, i32 40, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??0_System_error@std@@QEAA@AEBV01@@Z" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, section ".xdata", comdat
@"_CTA5?AVfailure@ios_base@std@@" = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.5 { i32 5, [5 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"_CT??_R0?AVfailure@ios_base@std@@@8??0failure@ios_base@std@@QEAA@AEBV012@@Z40" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"_CT??_R0?AVsystem_error@std@@@8??0system_error@std@@QEAA@AEBV01@@Z40" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"_CT??_R0?AV_System_error@std@@@8??0_System_error@std@@QEAA@AEBV01@@Z40" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"_CT??_R0?AVruntime_error@std@@@8??0runtime_error@std@@QEAA@AEBV01@@Z24" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"_CT??_R0?AVexception@std@@@8??0exception@std@@QEAA@AEBV01@@Z24" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32)] }, section ".xdata", comdat
@"_TI5?AVfailure@ios_base@std@@" = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??1failure@ios_base@std@@UEAA@XZ" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"_CTA5?AVfailure@ios_base@std@@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, section ".xdata", comdat
@"?_Static@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@1@XZ@4V21@B" = linkonce_odr dso_local constant { ptr, %"union.std::error_category::_Addr_storage" } { ptr @"??_7_Iostream_error_category2@std@@6B@", %"union.std::error_category::_Addr_storage" { i64 5 } }, comdat, align 8
@2 = private unnamed_addr constant { [7 x ptr] } { [7 x ptr] [ptr @"??_R4_Iostream_error_category2@std@@6B@", ptr @"??_G_Iostream_error_category2@std@@UEAAPEAXI@Z", ptr @"?name@_Iostream_error_category2@std@@UEBAPEBDXZ", ptr @"?message@_Iostream_error_category2@std@@UEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@H@Z", ptr @"?default_error_condition@error_category@std@@UEBA?AVerror_condition@2@H@Z", ptr @"?equivalent@error_category@std@@UEBA_NAEBVerror_code@2@H@Z", ptr @"?equivalent@error_category@std@@UEBA_NHAEBVerror_condition@2@@Z"] }, comdat($"??_7_Iostream_error_category2@std@@6B@")
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
@"?_Iostream_error@?4??message@_Iostream_error_category2@std@@UEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@3@H@Z@4QBDB" = linkonce_odr dso_local constant [22 x i8] c"iostream stream error\00", comdat, align 16
@"?_Fake_alloc@std@@3U_Fake_allocator@1@B" = linkonce_odr dso_local constant %"struct.std::_Fake_allocator" undef, comdat, align 1
@"??_C@_0BA@JFNIOLAK@string?5too?5long?$AA@" = linkonce_odr dso_local unnamed_addr constant [16 x i8] c"string too long\00", comdat, align 1
@"??_R0?AVbad_array_new_length@std@@@8" = linkonce_odr global %rtti.TypeDescriptor30 { ptr @"??_7type_info@@6B@", ptr null, [31 x i8] c".?AVbad_array_new_length@std@@\00" }, comdat
@"_CT??_R0?AVbad_array_new_length@std@@@8??0bad_array_new_length@std@@QEAA@AEBV01@@Z24" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVbad_array_new_length@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 -1, i32 0, i32 24, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??0bad_array_new_length@std@@QEAA@AEBV01@@Z" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, section ".xdata", comdat
@"??_R0?AVbad_alloc@std@@@8" = linkonce_odr global %rtti.TypeDescriptor19 { ptr @"??_7type_info@@6B@", ptr null, [20 x i8] c".?AVbad_alloc@std@@\00" }, comdat
@"_CT??_R0?AVbad_alloc@std@@@8??0bad_alloc@std@@QEAA@AEBV01@@Z24" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 16, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVbad_alloc@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 -1, i32 0, i32 24, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??0bad_alloc@std@@QEAA@AEBV01@@Z" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, section ".xdata", comdat
@"_CTA3?AVbad_array_new_length@std@@" = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.3 { i32 3, [3 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"_CT??_R0?AVbad_array_new_length@std@@@8??0bad_array_new_length@std@@QEAA@AEBV01@@Z24" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"_CT??_R0?AVbad_alloc@std@@@8??0bad_alloc@std@@QEAA@AEBV01@@Z24" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"_CT??_R0?AVexception@std@@@8??0exception@std@@QEAA@AEBV01@@Z24" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32)] }, section ".xdata", comdat
@"_TI3?AVbad_array_new_length@std@@" = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??1bad_array_new_length@std@@UEAA@XZ" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"_CTA3?AVbad_array_new_length@std@@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, section ".xdata", comdat
@"??_C@_0BF@KINCDENJ@bad?5array?5new?5length?$AA@" = linkonce_odr dso_local unnamed_addr constant [21 x i8] c"bad array new length\00", comdat, align 1
@3 = private unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr @"??_R4bad_array_new_length@std@@6B@", ptr @"??_Gbad_array_new_length@std@@UEAAPEAXI@Z", ptr @"?what@exception@std@@UEBAPEBDXZ"] }, comdat($"??_7bad_array_new_length@std@@6B@")
@"??_R4bad_array_new_length@std@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVbad_array_new_length@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3bad_array_new_length@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R4bad_array_new_length@std@@6B@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R3bad_array_new_length@std@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 0, i32 3, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R2bad_array_new_length@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R2bad_array_new_length@std@@8" = linkonce_odr constant [4 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@bad_array_new_length@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@bad_alloc@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@exception@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0], comdat
@"??_R1A@?0A@EA@bad_array_new_length@std@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVbad_array_new_length@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 2, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3bad_array_new_length@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R1A@?0A@EA@bad_alloc@std@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVbad_alloc@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 1, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3bad_alloc@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R3bad_alloc@std@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 0, i32 2, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R2bad_alloc@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R2bad_alloc@std@@8" = linkonce_odr constant [3 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@bad_alloc@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@exception@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0], comdat
@4 = private unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr @"??_R4bad_alloc@std@@6B@", ptr @"??_Gbad_alloc@std@@UEAAPEAXI@Z", ptr @"?what@exception@std@@UEBAPEBDXZ"] }, comdat($"??_7bad_alloc@std@@6B@")
@"??_R4bad_alloc@std@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVbad_alloc@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3bad_alloc@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R4bad_alloc@std@@6B@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@5 = private unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr @"??_R4failure@ios_base@std@@6B@", ptr @"??_Gfailure@ios_base@std@@UEAAPEAXI@Z", ptr @"?what@exception@std@@UEBAPEBDXZ"] }, comdat($"??_7failure@ios_base@std@@6B@")
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
@6 = private unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr @"??_R4system_error@std@@6B@", ptr @"??_Gsystem_error@std@@UEAAPEAXI@Z", ptr @"?what@exception@std@@UEBAPEBDXZ"] }, comdat($"??_7system_error@std@@6B@")
@"??_R4system_error@std@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AVsystem_error@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3system_error@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R4system_error@std@@6B@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@7 = private unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr @"??_R4_System_error@std@@6B@", ptr @"??_G_System_error@std@@UEAAPEAXI@Z", ptr @"?what@exception@std@@UEBAPEBDXZ"] }, comdat($"??_7_System_error@std@@6B@")
@"??_R4_System_error@std@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AV_System_error@std@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3_System_error@std@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R4_System_error@std@@6B@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_C@_02LMMGGCAJ@?3?5?$AA@" = linkonce_odr dso_local unnamed_addr constant [3 x i8] c": \00", comdat, align 1

@"??_7runtime_error@std@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [3 x ptr] }, ptr @0, i32 0, i32 0, i32 1)
@"??_7exception@std@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [3 x ptr] }, ptr @1, i32 0, i32 0, i32 1)
@"??_7_Iostream_error_category2@std@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [7 x ptr] }, ptr @2, i32 0, i32 0, i32 1)
@"??_7bad_array_new_length@std@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [3 x ptr] }, ptr @3, i32 0, i32 0, i32 1)
@"??_7bad_alloc@std@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [3 x ptr] }, ptr @4, i32 0, i32 0, i32 1)
@"??_7failure@ios_base@std@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [3 x ptr] }, ptr @5, i32 0, i32 0, i32 1)
@"??_7system_error@std@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [3 x ptr] }, ptr @6, i32 0, i32 0, i32 1)
@"??_7_System_error@std@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [3 x ptr] }, ptr @7, i32 0, i32 0, i32 1)

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define dso_local i64 @"?switch_to_new_thread@@YA@AEAVjthread@std@@@Z"(ptr noundef nonnull align 8 dereferenceable(24) %0) #0 {
  %2 = alloca %struct.awaitable, align 8
  %3 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  %4 = getelementptr inbounds nuw %struct.awaitable, ptr %2, i32 0, i32 0
  %5 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  store ptr %5, ptr %4, align 8
  %6 = getelementptr inbounds nuw %struct.awaitable, ptr %2, i32 0, i32 0
  %7 = load ptr, ptr %6, align 8
  %8 = ptrtoint ptr %7 to i64
  ret i64 %8
}

; Function Attrs: mustprogress noinline optnone presplitcoroutine sspstrong uwtable
define dso_local i8 @"?resuming_on_new_thread@@YA?AUtask@@AEAVjthread@std@@@Z"(ptr noundef nonnull align 8 dereferenceable(24) %0) #1 personality ptr @__CxxFrameHandler3 {
  %2 = alloca %struct.task, align 1
  %3 = alloca ptr, align 8, !coro.outside.frame !16
  %4 = alloca ptr, align 8
  %5 = alloca %"struct.task::promise_type", align 1
  %6 = alloca %"struct.std::suspend_never", align 1
  %7 = alloca %"class.std::thread::id", align 4
  %8 = alloca %struct.awaitable, align 8
  %9 = alloca %"class.std::thread::id", align 4
  %10 = alloca %"struct.std::suspend_never", align 1
  store ptr %0, ptr %3, align 8
  %11 = bitcast ptr %5 to ptr
  %12 = call token @llvm.coro.id(i32 16, ptr %11, ptr null, ptr null)
  %13 = call i1 @llvm.coro.alloc(token %12)
  br i1 %13, label %14, label %17

14:                                               ; preds = %1
  %15 = call i64 @llvm.coro.size.i64()
  %16 = call noalias noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef %15) #21
  br label %17

17:                                               ; preds = %14, %1
  %18 = phi ptr [ null, %1 ], [ %16, %14 ]
  %19 = call ptr @llvm.coro.begin(token %12, ptr %18)
  invoke void @llvm.seh.scope.begin()
          to label %20 unwind label %112

20:                                               ; preds = %17
  call void @llvm.lifetime.start.p0(i64 8, ptr %4) #3
  %21 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  store ptr %21, ptr %4, align 8
  call void @llvm.lifetime.start.p0(i64 1, ptr %5) #3
  invoke void @"?get_return_object@promise_type@task@@QEAA?AU2@XZ"(ptr noundef nonnull align 1 dereferenceable(1) %5, ptr dead_on_unwind writable sret(%struct.task) align 1 %2)
          to label %22 unwind label %108

22:                                               ; preds = %20
  invoke void @llvm.seh.scope.begin()
          to label %23 unwind label %104

23:                                               ; preds = %22
  call void @llvm.lifetime.start.p0(i64 1, ptr %6) #3
  invoke void @"?initial_suspend@promise_type@task@@QEAA?AUsuspend_never@std@@XZ"(ptr noundef nonnull align 1 dereferenceable(1) %5, ptr dead_on_unwind writable sret(%"struct.std::suspend_never") align 1 %6)
          to label %24 unwind label %54

24:                                               ; preds = %23
  %25 = call noundef zeroext i1 @"?await_ready@suspend_never@std@@QEBA_NXZ"(ptr noundef nonnull align 1 dereferenceable(1) %6) #3
  br i1 %25, label %30, label %26

26:                                               ; preds = %24
  %27 = call token @llvm.coro.save(ptr null)
  call void @llvm.coro.await.suspend.void(ptr %6, ptr %19, ptr @"?resuming_on_new_thread@@YA?AUtask@@AEAVjthread@std@@@Z.__await_suspend_wrapper__init") #3
  %28 = call i8 @llvm.coro.suspend(token %27, i1 false)
  switch i8 %28, label %100 [
    i8 0, label %30
    i8 1, label %29
  ]

29:                                               ; preds = %26
  br label %31

30:                                               ; preds = %26, %24
  call void @"?await_resume@suspend_never@std@@QEBAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %6) #3
  br label %31

31:                                               ; preds = %30, %29
  %32 = phi i32 [ 0, %30 ], [ 2, %29 ]
  call void @llvm.lifetime.end.p0(i64 1, ptr %6) #3
  switch i32 %32, label %91 [
    i32 0, label %33
  ]

33:                                               ; preds = %31
  invoke void @llvm.seh.try.begin()
          to label %34 unwind label %69

34:                                               ; preds = %33
  %35 = invoke noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(8) @"?cout@std@@3V?$basic_ostream@DU?$char_traits@D@std@@@1@A", ptr noundef @"??_C@_0BO@EKOPNEHI@Coroutine?5started?5on?5thread?3?5?$AA@")
          to label %36 unwind label %69

36:                                               ; preds = %34
  call void @"?get_id@this_thread@std@@YA?AVid@thread@2@XZ"(ptr dead_on_unwind writable sret(%"class.std::thread::id") align 4 %7) #3
  %37 = getelementptr inbounds nuw %"class.std::thread::id", ptr %7, i32 0, i32 0
  %38 = load i32, ptr %37, align 4
  %39 = invoke noundef nonnull align 8 dereferenceable(8) ptr @"??$?6DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@Vid@thread@0@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %35, i32 %38)
          to label %40 unwind label %69

40:                                               ; preds = %36
  %41 = invoke noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@D@Z"(ptr noundef nonnull align 8 dereferenceable(8) %39, i8 noundef 10)
          to label %42 unwind label %69

42:                                               ; preds = %40
  call void @llvm.lifetime.start.p0(i64 8, ptr %8) #3
  %43 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %44 = call i64 @"?switch_to_new_thread@@YA@AEAVjthread@std@@@Z"(ptr noundef nonnull align 8 dereferenceable(24) %43)
  %45 = getelementptr inbounds nuw %struct.awaitable, ptr %8, i32 0, i32 0
  %46 = inttoptr i64 %44 to ptr
  store ptr %46, ptr %45, align 8
  %47 = invoke noundef zeroext i1 @"?await_ready@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAA_NXZ"(ptr noundef nonnull align 8 dereferenceable(8) %8)
          to label %48 unwind label %84

48:                                               ; preds = %42
  br i1 %47, label %56, label %49

49:                                               ; preds = %48
  %50 = call token @llvm.coro.save(ptr null)
  invoke void @llvm.coro.await.suspend.void(ptr %8, ptr %19, ptr @"?resuming_on_new_thread@@YA?AUtask@@AEAVjthread@std@@@Z.__await_suspend_wrapper__await")
          to label %51 unwind label %84

51:                                               ; preds = %49
  %52 = call i8 @llvm.coro.suspend(token %50, i1 false)
  switch i8 %52, label %100 [
    i8 0, label %56
    i8 1, label %53
  ]

53:                                               ; preds = %51
  br label %58

54:                                               ; preds = %23
  %55 = cleanuppad within none []
  call void @llvm.lifetime.end.p0(i64 1, ptr %6) #3
  cleanupret from %55 unwind label %104

56:                                               ; preds = %51, %48
  invoke void @"?await_resume@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXXZ"(ptr noundef nonnull align 8 dereferenceable(8) %8)
          to label %57 unwind label %84

57:                                               ; preds = %56
  br label %58

58:                                               ; preds = %57, %53
  %59 = phi i32 [ 0, %57 ], [ 2, %53 ]
  call void @llvm.lifetime.end.p0(i64 8, ptr %8) #3
  switch i32 %59, label %91 [
    i32 0, label %60
  ]

60:                                               ; preds = %58
  %61 = invoke noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(8) @"?cout@std@@3V?$basic_ostream@DU?$char_traits@D@std@@@1@A", ptr noundef @"??_C@_0BO@NLJBCPPM@Coroutine?5resumed?5on?5thread?3?5?$AA@")
          to label %62 unwind label %69

62:                                               ; preds = %60
  call void @"?get_id@this_thread@std@@YA?AVid@thread@2@XZ"(ptr dead_on_unwind writable sret(%"class.std::thread::id") align 4 %9) #3
  %63 = getelementptr inbounds nuw %"class.std::thread::id", ptr %9, i32 0, i32 0
  %64 = load i32, ptr %63, align 4
  %65 = invoke noundef nonnull align 8 dereferenceable(8) ptr @"??$?6DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@Vid@thread@0@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %61, i32 %64)
          to label %66 unwind label %69

66:                                               ; preds = %62
  %67 = invoke noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@D@Z"(ptr noundef nonnull align 8 dereferenceable(8) %65, i8 noundef 10)
          to label %68 unwind label %69

68:                                               ; preds = %66
  invoke void @"?return_void@promise_type@task@@QEAAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %5)
          to label %83 unwind label %69

69:                                               ; preds = %68, %66, %62, %60, %84, %40, %36, %34, %33
  %70 = catchswitch within none [label %71] unwind label %104

71:                                               ; preds = %69
  %72 = catchpad within %70 [ptr null, i32 0, ptr null]
  invoke void @"?unhandled_exception@promise_type@task@@QEAAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %5) [ "funclet"(token %72) ]
          to label %73 unwind label %104

73:                                               ; preds = %71
  invoke void @llvm.seh.scope.end() [ "funclet"(token %72) ]
          to label %74 unwind label %104

74:                                               ; preds = %73
  catchret from %72 to label %75

75:                                               ; preds = %74
  br label %76

76:                                               ; preds = %75
  br label %77

77:                                               ; preds = %76, %83
  call void @llvm.lifetime.start.p0(i64 1, ptr %10) #3
  call void @"?final_suspend@promise_type@task@@QEAA?AUsuspend_never@std@@XZ"(ptr noundef nonnull align 1 dereferenceable(1) %5, ptr dead_on_unwind writable sret(%"struct.std::suspend_never") align 1 %10) #3
  %78 = call noundef zeroext i1 @"?await_ready@suspend_never@std@@QEBA_NXZ"(ptr noundef nonnull align 1 dereferenceable(1) %10) #3
  br i1 %78, label %86, label %79

79:                                               ; preds = %77
  %80 = call token @llvm.coro.save(ptr null)
  call void @llvm.coro.await.suspend.void(ptr %10, ptr %19, ptr @"?resuming_on_new_thread@@YA?AUtask@@AEAVjthread@std@@@Z.__await_suspend_wrapper__final") #3
  %81 = call i8 @llvm.coro.suspend(token %80, i1 true)
  switch i8 %81, label %100 [
    i8 0, label %86
    i8 1, label %82
  ]

82:                                               ; preds = %79
  br label %87

83:                                               ; preds = %68
  br label %77

84:                                               ; preds = %56, %49, %42
  %85 = cleanuppad within none []
  call void @llvm.lifetime.end.p0(i64 8, ptr %8) #3
  cleanupret from %85 unwind label %69

86:                                               ; preds = %79, %77
  call void @"?await_resume@suspend_never@std@@QEBAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %10) #3
  br label %87

87:                                               ; preds = %86, %82
  %88 = phi i32 [ 0, %86 ], [ 2, %82 ]
  call void @llvm.lifetime.end.p0(i64 1, ptr %10) #3
  switch i32 %88, label %91 [
    i32 0, label %89
  ]

89:                                               ; preds = %87
  invoke void @llvm.seh.scope.end()
          to label %90 unwind label %104

90:                                               ; preds = %89
  br label %91

91:                                               ; preds = %90, %87, %58, %31
  %92 = phi i32 [ %32, %31 ], [ %59, %58 ], [ %88, %87 ], [ 0, %90 ]
  call void @llvm.lifetime.end.p0(i64 1, ptr %5) #3
  call void @llvm.lifetime.end.p0(i64 8, ptr %4) #3
  invoke void @llvm.seh.scope.end()
          to label %93 unwind label %112

93:                                               ; preds = %91
  %94 = call ptr @llvm.coro.free(token %12, ptr %19)
  %95 = icmp ne ptr %94, null
  br i1 %95, label %96, label %98

96:                                               ; preds = %93
  %97 = call i64 @llvm.coro.size.i64()
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %94, i64 noundef %97) #3
  br label %98

98:                                               ; preds = %93, %96
  switch i32 %92, label %119 [
    i32 0, label %99
    i32 2, label %100
  ]

99:                                               ; preds = %98
  br label %100

100:                                              ; preds = %99, %98, %79, %51, %26
  %101 = call i1 @llvm.coro.end(ptr null, i1 false, token none)
  %102 = getelementptr inbounds nuw %struct.task, ptr %2, i32 0, i32 0
  %103 = load i8, ptr %102, align 1
  ret i8 %103

104:                                              ; preds = %89, %73, %71, %69, %54, %22
  %105 = cleanuppad within none []
  invoke void @llvm.seh.scope.end() [ "funclet"(token %105) ]
          to label %106 unwind label %108

106:                                              ; preds = %104
  %107 = call i1 @llvm.coro.end(ptr null, i1 true, token none) [ "funclet"(token %105) ]
  cleanupret from %105 unwind label %108

108:                                              ; preds = %106, %104, %20
  %109 = cleanuppad within none []
  call void @llvm.lifetime.end.p0(i64 1, ptr %5) #3
  cleanupret from %109 unwind label %110

110:                                              ; preds = %108
  %111 = cleanuppad within none []
  call void @llvm.lifetime.end.p0(i64 8, ptr %4) #3
  cleanupret from %111 unwind label %112

112:                                              ; preds = %91, %110, %17
  %113 = cleanuppad within none []
  %114 = call ptr @llvm.coro.free(token %12, ptr %19)
  %115 = icmp ne ptr %114, null
  br i1 %115, label %116, label %118

116:                                              ; preds = %112
  %117 = call i64 @llvm.coro.size.i64()
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %114, i64 noundef %117) #3 [ "funclet"(token %113) ]
  br label %118

118:                                              ; preds = %112, %116
  cleanupret from %113 unwind to caller

119:                                              ; preds = %98
  unreachable
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare token @llvm.coro.id(i32, ptr readnone, ptr readonly captures(none), ptr) #2

; Function Attrs: nounwind
declare i1 @llvm.coro.alloc(token) #3

; Function Attrs: nobuiltin allocsize(0)
declare dso_local noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef) #4

; Function Attrs: nounwind memory(none)
declare i64 @llvm.coro.size.i64() #5

; Function Attrs: nounwind
declare ptr @llvm.coro.begin(token, ptr writeonly) #3

declare dso_local i32 @__CxxFrameHandler3(...)

; Function Attrs: nounwind memory(none)
declare dso_local void @llvm.seh.scope.begin() #5

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #6

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?get_return_object@promise_type@task@@QEAA?AU2@XZ"(ptr noundef nonnull align 1 dereferenceable(1) %0, ptr dead_on_unwind noalias writable sret(%struct.task) align 1 %1) #0 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?initial_suspend@promise_type@task@@QEAA?AUsuspend_never@std@@XZ"(ptr noundef nonnull align 1 dereferenceable(1) %0, ptr dead_on_unwind noalias writable sret(%"struct.std::suspend_never") align 1 %1) #0 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef zeroext i1 @"?await_ready@suspend_never@std@@QEBA_NXZ"(ptr noundef nonnull align 1 dereferenceable(1) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret i1 true
}

; Function Attrs: nomerge nounwind
declare token @llvm.coro.save(ptr) #7

; Function Attrs: alwaysinline mustprogress
define private void @"?resuming_on_new_thread@@YA?AUtask@@AEAVjthread@std@@@Z.__await_suspend_wrapper__init"(ptr noundef nonnull %0, ptr noundef %1) #8 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca %"struct.std::coroutine_handle", align 8
  %6 = alloca %"struct.std::coroutine_handle.0", align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %7 = load ptr, ptr %4, align 8
  %8 = load ptr, ptr %3, align 8
  call void @"?from_address@?$coroutine_handle@Upromise_type@task@@@std@@SA?AU12@QEAX@Z"(ptr dead_on_unwind writable sret(%"struct.std::coroutine_handle.0") align 8 %6, ptr noundef %8) #3
  call void @"??B?$coroutine_handle@Upromise_type@task@@@std@@QEBA?AU?$coroutine_handle@X@1@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %6, ptr dead_on_unwind writable sret(%"struct.std::coroutine_handle") align 8 %5) #3
  %9 = getelementptr inbounds nuw %"struct.std::coroutine_handle", ptr %5, i32 0, i32 0
  %10 = load ptr, ptr %9, align 8
  %11 = ptrtoint ptr %10 to i64
  call void @"?await_suspend@suspend_never@std@@QEBAXU?$coroutine_handle@X@2@@Z"(ptr noundef nonnull align 1 dereferenceable(1) %7, i64 %11) #3
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?await_suspend@suspend_never@std@@QEBAXU?$coroutine_handle@X@2@@Z"(ptr noundef nonnull align 1 dereferenceable(1) %0, i64 %1) #0 comdat align 2 {
  %3 = alloca %"struct.std::coroutine_handle", align 8
  %4 = alloca ptr, align 8
  %5 = getelementptr inbounds nuw %"struct.std::coroutine_handle", ptr %3, i32 0, i32 0
  %6 = inttoptr i64 %1 to ptr
  store ptr %6, ptr %5, align 8
  store ptr %0, ptr %4, align 8
  %7 = load ptr, ptr %4, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?from_address@?$coroutine_handle@Upromise_type@task@@@std@@SA?AU12@QEAX@Z"(ptr dead_on_unwind noalias writable sret(%"struct.std::coroutine_handle.0") align 8 %0, ptr noundef %1) #0 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %5 = call noundef ptr @"??0?$coroutine_handle@Upromise_type@task@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) #3
  %6 = load ptr, ptr %4, align 8
  %7 = getelementptr inbounds nuw %"struct.std::coroutine_handle.0", ptr %0, i32 0, i32 0
  store ptr %6, ptr %7, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"??B?$coroutine_handle@Upromise_type@task@@@std@@QEBA?AU?$coroutine_handle@X@1@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %0, ptr dead_on_unwind noalias writable sret(%"struct.std::coroutine_handle") align 8 %1) #0 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr inbounds nuw %"struct.std::coroutine_handle.0", ptr %5, i32 0, i32 0
  %7 = load ptr, ptr %6, align 8
  call void @"?from_address@?$coroutine_handle@X@std@@SA?AU12@QEAX@Z"(ptr dead_on_unwind writable sret(%"struct.std::coroutine_handle") align 8 %1, ptr noundef %7) #3
  ret void
}

declare void @llvm.coro.await.suspend.void(ptr, ptr, ptr)

; Function Attrs: nounwind
declare i8 @llvm.coro.suspend(token, i1) #3

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?await_resume@suspend_never@std@@QEBAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr captures(none)) #6

; Function Attrs: nounwind willreturn memory(write)
declare dso_local void @llvm.seh.try.begin() #9

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@D@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0, i8 noundef %1) #10 comdat personality ptr @__CxxFrameHandler3 {
  %3 = alloca i8, align 1
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  %6 = alloca %"class.std::basic_ostream<char>::sentry", align 8
  %7 = alloca i64, align 8
  store i8 %1, ptr %3, align 1
  store ptr %0, ptr %4, align 8
  store i32 0, ptr %5, align 4
  %8 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %9 = call noundef ptr @"??0sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z"(ptr noundef nonnull align 8 dereferenceable(16) %6, ptr noundef nonnull align 8 dereferenceable(8) %8)
  invoke void @llvm.seh.scope.begin()
          to label %10 unwind label %188

10:                                               ; preds = %2
  %11 = call noundef zeroext i1 @"??Bsentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEBA_NXZ"(ptr noundef nonnull align 8 dereferenceable(16) %6)
  br i1 %11, label %12, label %166

12:                                               ; preds = %10
  %13 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %14 = getelementptr inbounds i8, ptr %13, i64 0
  %15 = load ptr, ptr %14, align 8
  %16 = getelementptr inbounds i32, ptr %15, i32 1
  %17 = load i32, ptr %16, align 4
  %18 = sext i32 %17 to i64
  %19 = add nsw i64 0, %18
  %20 = getelementptr inbounds i8, ptr %13, i64 %19
  %21 = call noundef i64 @"?width@ios_base@std@@QEBA_JXZ"(ptr noundef nonnull align 8 dereferenceable(72) %20) #3
  %22 = icmp sle i64 %21, 1
  br i1 %22, label %23, label %24

23:                                               ; preds = %12
  br label %35

24:                                               ; preds = %12
  %25 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %26 = getelementptr inbounds i8, ptr %25, i64 0
  %27 = load ptr, ptr %26, align 8
  %28 = getelementptr inbounds i32, ptr %27, i32 1
  %29 = load i32, ptr %28, align 4
  %30 = sext i32 %29 to i64
  %31 = add nsw i64 0, %30
  %32 = getelementptr inbounds i8, ptr %25, i64 %31
  %33 = call noundef i64 @"?width@ios_base@std@@QEBA_JXZ"(ptr noundef nonnull align 8 dereferenceable(72) %32) #3
  %34 = sub nsw i64 %33, 1
  br label %35

35:                                               ; preds = %24, %23
  %36 = phi i64 [ 0, %23 ], [ %34, %24 ]
  store i64 %36, ptr %7, align 8
  invoke void @llvm.seh.try.begin()
          to label %37 unwind label %139

37:                                               ; preds = %35
  %38 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %39 = getelementptr inbounds i8, ptr %38, i64 0
  %40 = load ptr, ptr %39, align 8
  %41 = getelementptr inbounds i32, ptr %40, i32 1
  %42 = load i32, ptr %41, align 4
  %43 = sext i32 %42 to i64
  %44 = add nsw i64 0, %43
  %45 = getelementptr inbounds i8, ptr %38, i64 %44
  %46 = call noundef i32 @"?flags@ios_base@std@@QEBAHXZ"(ptr noundef nonnull align 8 dereferenceable(72) %45) #3
  %47 = and i32 %46, 448
  %48 = icmp ne i32 %47, 64
  br i1 %48, label %49, label %89

49:                                               ; preds = %37
  br label %50

50:                                               ; preds = %85, %49
  %51 = load i32, ptr %5, align 4
  %52 = icmp eq i32 %51, 0
  br i1 %52, label %53, label %56

53:                                               ; preds = %50
  %54 = load i64, ptr %7, align 8
  %55 = icmp slt i64 0, %54
  br label %56

56:                                               ; preds = %53, %50
  %57 = phi i1 [ false, %50 ], [ %55, %53 ]
  br i1 %57, label %58, label %88

58:                                               ; preds = %56
  %59 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %60 = getelementptr inbounds i8, ptr %59, i64 0
  %61 = load ptr, ptr %60, align 8
  %62 = getelementptr inbounds i32, ptr %61, i32 1
  %63 = load i32, ptr %62, align 4
  %64 = sext i32 %63 to i64
  %65 = add nsw i64 0, %64
  %66 = getelementptr inbounds i8, ptr %59, i64 %65
  %67 = call noundef ptr @"?rdbuf@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_streambuf@DU?$char_traits@D@std@@@2@XZ"(ptr noundef nonnull align 8 dereferenceable(96) %66) #3
  %68 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %69 = getelementptr inbounds i8, ptr %68, i64 0
  %70 = load ptr, ptr %69, align 8
  %71 = getelementptr inbounds i32, ptr %70, i32 1
  %72 = load i32, ptr %71, align 4
  %73 = sext i32 %72 to i64
  %74 = add nsw i64 0, %73
  %75 = getelementptr inbounds i8, ptr %68, i64 %74
  %76 = call noundef i8 @"?fill@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADXZ"(ptr noundef nonnull align 8 dereferenceable(96) %75) #3
  %77 = invoke noundef i32 @"?sputc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHD@Z"(ptr noundef nonnull align 8 dereferenceable(104) %67, i8 noundef %76)
          to label %78 unwind label %139

78:                                               ; preds = %58
  %79 = call noundef i32 @"?eof@?$_Narrow_char_traits@DH@std@@SAHXZ"() #3
  %80 = call noundef zeroext i1 @"?eq_int_type@?$_Narrow_char_traits@DH@std@@SA_NHH@Z"(i32 noundef %79, i32 noundef %77) #3
  br i1 %80, label %81, label %84

81:                                               ; preds = %78
  %82 = load i32, ptr %5, align 4
  %83 = or i32 %82, 4
  store i32 %83, ptr %5, align 4
  br label %84

84:                                               ; preds = %81, %78
  br label %85

85:                                               ; preds = %84
  %86 = load i64, ptr %7, align 8
  %87 = add nsw i64 %86, -1
  store i64 %87, ptr %7, align 8
  br label %50, !llvm.loop !18

88:                                               ; preds = %56
  br label %89

89:                                               ; preds = %88, %37
  %90 = load i32, ptr %5, align 4
  %91 = icmp eq i32 %90, 0
  br i1 %91, label %92, label %110

92:                                               ; preds = %89
  %93 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %94 = getelementptr inbounds i8, ptr %93, i64 0
  %95 = load ptr, ptr %94, align 8
  %96 = getelementptr inbounds i32, ptr %95, i32 1
  %97 = load i32, ptr %96, align 4
  %98 = sext i32 %97 to i64
  %99 = add nsw i64 0, %98
  %100 = getelementptr inbounds i8, ptr %93, i64 %99
  %101 = call noundef ptr @"?rdbuf@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_streambuf@DU?$char_traits@D@std@@@2@XZ"(ptr noundef nonnull align 8 dereferenceable(96) %100) #3
  %102 = load i8, ptr %3, align 1
  %103 = invoke noundef i32 @"?sputc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHD@Z"(ptr noundef nonnull align 8 dereferenceable(104) %101, i8 noundef %102)
          to label %104 unwind label %139

104:                                              ; preds = %92
  %105 = call noundef i32 @"?eof@?$_Narrow_char_traits@DH@std@@SAHXZ"() #3
  %106 = call noundef zeroext i1 @"?eq_int_type@?$_Narrow_char_traits@DH@std@@SA_NHH@Z"(i32 noundef %105, i32 noundef %103) #3
  br i1 %106, label %107, label %110

107:                                              ; preds = %104
  %108 = load i32, ptr %5, align 4
  %109 = or i32 %108, 4
  store i32 %109, ptr %5, align 4
  br label %110

110:                                              ; preds = %107, %104, %89
  br label %111

111:                                              ; preds = %162, %110
  %112 = load i32, ptr %5, align 4
  %113 = icmp eq i32 %112, 0
  br i1 %113, label %114, label %117

114:                                              ; preds = %111
  %115 = load i64, ptr %7, align 8
  %116 = icmp slt i64 0, %115
  br label %117

117:                                              ; preds = %114, %111
  %118 = phi i1 [ false, %111 ], [ %116, %114 ]
  br i1 %118, label %119, label %165

119:                                              ; preds = %117
  %120 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %121 = getelementptr inbounds i8, ptr %120, i64 0
  %122 = load ptr, ptr %121, align 8
  %123 = getelementptr inbounds i32, ptr %122, i32 1
  %124 = load i32, ptr %123, align 4
  %125 = sext i32 %124 to i64
  %126 = add nsw i64 0, %125
  %127 = getelementptr inbounds i8, ptr %120, i64 %126
  %128 = call noundef ptr @"?rdbuf@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_streambuf@DU?$char_traits@D@std@@@2@XZ"(ptr noundef nonnull align 8 dereferenceable(96) %127) #3
  %129 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %130 = getelementptr inbounds i8, ptr %129, i64 0
  %131 = load ptr, ptr %130, align 8
  %132 = getelementptr inbounds i32, ptr %131, i32 1
  %133 = load i32, ptr %132, align 4
  %134 = sext i32 %133 to i64
  %135 = add nsw i64 0, %134
  %136 = getelementptr inbounds i8, ptr %129, i64 %135
  %137 = call noundef i8 @"?fill@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADXZ"(ptr noundef nonnull align 8 dereferenceable(96) %136) #3
  %138 = invoke noundef i32 @"?sputc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHD@Z"(ptr noundef nonnull align 8 dereferenceable(104) %128, i8 noundef %137)
          to label %155 unwind label %139

139:                                              ; preds = %119, %92, %58, %35
  %140 = catchswitch within none [label %141] unwind label %188

141:                                              ; preds = %139
  %142 = catchpad within %140 [ptr null, i32 0, ptr null]
  %143 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %144 = getelementptr inbounds i8, ptr %143, i64 0
  %145 = load ptr, ptr %144, align 8
  %146 = getelementptr inbounds i32, ptr %145, i32 1
  %147 = load i32, ptr %146, align 4
  %148 = sext i32 %147 to i64
  %149 = add nsw i64 0, %148
  %150 = getelementptr inbounds i8, ptr %143, i64 %149
  invoke void @"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"(ptr noundef nonnull align 8 dereferenceable(96) %150, i32 noundef 4, i1 noundef zeroext true) [ "funclet"(token %142) ]
          to label %151 unwind label %188

151:                                              ; preds = %141
  invoke void @llvm.seh.scope.end() [ "funclet"(token %142) ]
          to label %152 unwind label %188

152:                                              ; preds = %151
  catchret from %142 to label %153

153:                                              ; preds = %152
  br label %154

154:                                              ; preds = %153, %165
  br label %166

155:                                              ; preds = %119
  %156 = call noundef i32 @"?eof@?$_Narrow_char_traits@DH@std@@SAHXZ"() #3
  %157 = call noundef zeroext i1 @"?eq_int_type@?$_Narrow_char_traits@DH@std@@SA_NHH@Z"(i32 noundef %156, i32 noundef %138) #3
  br i1 %157, label %158, label %161

158:                                              ; preds = %155
  %159 = load i32, ptr %5, align 4
  %160 = or i32 %159, 4
  store i32 %160, ptr %5, align 4
  br label %161

161:                                              ; preds = %158, %155
  br label %162

162:                                              ; preds = %161
  %163 = load i64, ptr %7, align 8
  %164 = add nsw i64 %163, -1
  store i64 %164, ptr %7, align 8
  br label %111, !llvm.loop !20

165:                                              ; preds = %117
  br label %154

166:                                              ; preds = %154, %10
  %167 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %168 = getelementptr inbounds i8, ptr %167, i64 0
  %169 = load ptr, ptr %168, align 8
  %170 = getelementptr inbounds i32, ptr %169, i32 1
  %171 = load i32, ptr %170, align 4
  %172 = sext i32 %171 to i64
  %173 = add nsw i64 0, %172
  %174 = getelementptr inbounds i8, ptr %167, i64 %173
  %175 = call noundef i64 @"?width@ios_base@std@@QEAA_J_J@Z"(ptr noundef nonnull align 8 dereferenceable(72) %174, i64 noundef 0) #3
  %176 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %177 = getelementptr inbounds i8, ptr %176, i64 0
  %178 = load ptr, ptr %177, align 8
  %179 = getelementptr inbounds i32, ptr %178, i32 1
  %180 = load i32, ptr %179, align 4
  %181 = sext i32 %180 to i64
  %182 = add nsw i64 0, %181
  %183 = getelementptr inbounds i8, ptr %176, i64 %182
  %184 = load i32, ptr %5, align 4
  invoke void @"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"(ptr noundef nonnull align 8 dereferenceable(96) %183, i32 noundef %184, i1 noundef zeroext false)
          to label %185 unwind label %188

185:                                              ; preds = %166
  %186 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  invoke void @llvm.seh.scope.end()
          to label %187 unwind label %188

187:                                              ; preds = %185
  call void @"??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %6) #3
  ret ptr %186

188:                                              ; preds = %185, %166, %151, %141, %139, %2
  %189 = cleanuppad within none []
  call void @"??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %6) #3 [ "funclet"(token %189) ]
  cleanupret from %189 unwind to caller
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(8) ptr @"??$?6DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@Vid@thread@0@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0, i32 %1) #10 comdat {
  %3 = alloca %"class.std::thread::id", align 4
  %4 = alloca ptr, align 8
  %5 = alloca [11 x i8], align 1
  %6 = alloca ptr, align 8
  %7 = getelementptr inbounds nuw %"class.std::thread::id", ptr %3, i32 0, i32 0
  store i32 %1, ptr %7, align 4
  store ptr %0, ptr %4, align 8
  %8 = call noundef ptr @"??$end@D$0L@@std@@YAPEADAEAY0L@D@Z"(ptr noundef nonnull align 1 dereferenceable(11) %5) #3
  store ptr %8, ptr %6, align 8
  %9 = load ptr, ptr %6, align 8
  %10 = getelementptr inbounds i8, ptr %9, i32 -1
  store ptr %10, ptr %6, align 8
  store i8 0, ptr %10, align 1
  %11 = getelementptr inbounds nuw %"class.std::thread::id", ptr %3, i32 0, i32 0
  %12 = load i32, ptr %11, align 4
  %13 = load ptr, ptr %6, align 8
  %14 = call noundef ptr @"??$_UIntegral_to_buff@DI@std@@YAPEADPEADI@Z"(ptr noundef %13, i32 noundef %12)
  store ptr %14, ptr %6, align 8
  %15 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %16 = load ptr, ptr %6, align 8
  %17 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(8) %15, ptr noundef %16)
  ret ptr %17
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0, ptr noundef %1) #10 comdat personality ptr @__CxxFrameHandler3 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  %6 = alloca i64, align 8
  %7 = alloca i64, align 8
  %8 = alloca %"class.std::basic_ostream<char>::sentry", align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  store i32 0, ptr %5, align 4
  %9 = load ptr, ptr %3, align 8
  %10 = call noundef i64 @"?length@?$_Narrow_char_traits@DH@std@@SA_KQEBD@Z"(ptr noundef %9) #3
  store i64 %10, ptr %6, align 8
  %11 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %12 = getelementptr inbounds i8, ptr %11, i64 0
  %13 = load ptr, ptr %12, align 8
  %14 = getelementptr inbounds i32, ptr %13, i32 1
  %15 = load i32, ptr %14, align 4
  %16 = sext i32 %15 to i64
  %17 = add nsw i64 0, %16
  %18 = getelementptr inbounds i8, ptr %11, i64 %17
  %19 = call noundef i64 @"?width@ios_base@std@@QEBA_JXZ"(ptr noundef nonnull align 8 dereferenceable(72) %18) #3
  %20 = icmp sle i64 %19, 0
  br i1 %20, label %33, label %21

21:                                               ; preds = %2
  %22 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %23 = getelementptr inbounds i8, ptr %22, i64 0
  %24 = load ptr, ptr %23, align 8
  %25 = getelementptr inbounds i32, ptr %24, i32 1
  %26 = load i32, ptr %25, align 4
  %27 = sext i32 %26 to i64
  %28 = add nsw i64 0, %27
  %29 = getelementptr inbounds i8, ptr %22, i64 %28
  %30 = call noundef i64 @"?width@ios_base@std@@QEBA_JXZ"(ptr noundef nonnull align 8 dereferenceable(72) %29) #3
  %31 = load i64, ptr %6, align 8
  %32 = icmp sle i64 %30, %31
  br i1 %32, label %33, label %34

33:                                               ; preds = %21, %2
  br label %46

34:                                               ; preds = %21
  %35 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %36 = getelementptr inbounds i8, ptr %35, i64 0
  %37 = load ptr, ptr %36, align 8
  %38 = getelementptr inbounds i32, ptr %37, i32 1
  %39 = load i32, ptr %38, align 4
  %40 = sext i32 %39 to i64
  %41 = add nsw i64 0, %40
  %42 = getelementptr inbounds i8, ptr %35, i64 %41
  %43 = call noundef i64 @"?width@ios_base@std@@QEBA_JXZ"(ptr noundef nonnull align 8 dereferenceable(72) %42) #3
  %44 = load i64, ptr %6, align 8
  %45 = sub nsw i64 %43, %44
  br label %46

46:                                               ; preds = %34, %33
  %47 = phi i64 [ 0, %33 ], [ %45, %34 ]
  store i64 %47, ptr %7, align 8
  %48 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %49 = call noundef ptr @"??0sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z"(ptr noundef nonnull align 8 dereferenceable(16) %8, ptr noundef nonnull align 8 dereferenceable(8) %48)
  invoke void @llvm.seh.scope.begin()
          to label %50 unwind label %203

50:                                               ; preds = %46
  %51 = invoke noundef zeroext i1 @"??Bsentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEBA_NXZ"(ptr noundef nonnull align 8 dereferenceable(16) %8)
          to label %52 unwind label %203

52:                                               ; preds = %50
  br i1 %51, label %56, label %53

53:                                               ; preds = %52
  %54 = load i32, ptr %5, align 4
  %55 = or i32 %54, 4
  store i32 %55, ptr %5, align 4
  br label %190

56:                                               ; preds = %52
  invoke void @llvm.seh.try.begin()
          to label %57 unwind label %153

57:                                               ; preds = %56
  %58 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %59 = getelementptr inbounds i8, ptr %58, i64 0
  %60 = load ptr, ptr %59, align 8
  %61 = getelementptr inbounds i32, ptr %60, i32 1
  %62 = load i32, ptr %61, align 4
  %63 = sext i32 %62 to i64
  %64 = add nsw i64 0, %63
  %65 = getelementptr inbounds i8, ptr %58, i64 %64
  %66 = call noundef i32 @"?flags@ios_base@std@@QEBAHXZ"(ptr noundef nonnull align 8 dereferenceable(72) %65) #3
  %67 = and i32 %66, 448
  %68 = icmp ne i32 %67, 64
  br i1 %68, label %69, label %104

69:                                               ; preds = %57
  br label %70

70:                                               ; preds = %100, %69
  %71 = load i64, ptr %7, align 8
  %72 = icmp slt i64 0, %71
  br i1 %72, label %73, label %103

73:                                               ; preds = %70
  %74 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %75 = getelementptr inbounds i8, ptr %74, i64 0
  %76 = load ptr, ptr %75, align 8
  %77 = getelementptr inbounds i32, ptr %76, i32 1
  %78 = load i32, ptr %77, align 4
  %79 = sext i32 %78 to i64
  %80 = add nsw i64 0, %79
  %81 = getelementptr inbounds i8, ptr %74, i64 %80
  %82 = call noundef ptr @"?rdbuf@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_streambuf@DU?$char_traits@D@std@@@2@XZ"(ptr noundef nonnull align 8 dereferenceable(96) %81) #3
  %83 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %84 = getelementptr inbounds i8, ptr %83, i64 0
  %85 = load ptr, ptr %84, align 8
  %86 = getelementptr inbounds i32, ptr %85, i32 1
  %87 = load i32, ptr %86, align 4
  %88 = sext i32 %87 to i64
  %89 = add nsw i64 0, %88
  %90 = getelementptr inbounds i8, ptr %83, i64 %89
  %91 = call noundef i8 @"?fill@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADXZ"(ptr noundef nonnull align 8 dereferenceable(96) %90) #3
  %92 = invoke noundef i32 @"?sputc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHD@Z"(ptr noundef nonnull align 8 dereferenceable(104) %82, i8 noundef %91)
          to label %93 unwind label %153

93:                                               ; preds = %73
  %94 = call noundef i32 @"?eof@?$_Narrow_char_traits@DH@std@@SAHXZ"() #3
  %95 = call noundef zeroext i1 @"?eq_int_type@?$_Narrow_char_traits@DH@std@@SA_NHH@Z"(i32 noundef %94, i32 noundef %92) #3
  br i1 %95, label %96, label %99

96:                                               ; preds = %93
  %97 = load i32, ptr %5, align 4
  %98 = or i32 %97, 4
  store i32 %98, ptr %5, align 4
  br label %103

99:                                               ; preds = %93
  br label %100

100:                                              ; preds = %99
  %101 = load i64, ptr %7, align 8
  %102 = add nsw i64 %101, -1
  store i64 %102, ptr %7, align 8
  br label %70, !llvm.loop !21

103:                                              ; preds = %96, %70
  br label %104

104:                                              ; preds = %103, %57
  %105 = load i32, ptr %5, align 4
  %106 = icmp eq i32 %105, 0
  br i1 %106, label %107, label %126

107:                                              ; preds = %104
  %108 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %109 = getelementptr inbounds i8, ptr %108, i64 0
  %110 = load ptr, ptr %109, align 8
  %111 = getelementptr inbounds i32, ptr %110, i32 1
  %112 = load i32, ptr %111, align 4
  %113 = sext i32 %112 to i64
  %114 = add nsw i64 0, %113
  %115 = getelementptr inbounds i8, ptr %108, i64 %114
  %116 = call noundef ptr @"?rdbuf@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_streambuf@DU?$char_traits@D@std@@@2@XZ"(ptr noundef nonnull align 8 dereferenceable(96) %115) #3
  %117 = load i64, ptr %6, align 8
  %118 = load ptr, ptr %3, align 8
  %119 = invoke noundef i64 @"?sputn@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAA_JPEBD_J@Z"(ptr noundef nonnull align 8 dereferenceable(104) %116, ptr noundef %118, i64 noundef %117)
          to label %120 unwind label %153

120:                                              ; preds = %107
  %121 = load i64, ptr %6, align 8
  %122 = icmp ne i64 %119, %121
  br i1 %122, label %123, label %126

123:                                              ; preds = %120
  %124 = load i32, ptr %5, align 4
  %125 = or i32 %124, 4
  store i32 %125, ptr %5, align 4
  br label %126

126:                                              ; preds = %123, %120, %104
  %127 = load i32, ptr %5, align 4
  %128 = icmp eq i32 %127, 0
  br i1 %128, label %129, label %180

129:                                              ; preds = %126
  br label %130

130:                                              ; preds = %176, %129
  %131 = load i64, ptr %7, align 8
  %132 = icmp slt i64 0, %131
  br i1 %132, label %133, label %179

133:                                              ; preds = %130
  %134 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %135 = getelementptr inbounds i8, ptr %134, i64 0
  %136 = load ptr, ptr %135, align 8
  %137 = getelementptr inbounds i32, ptr %136, i32 1
  %138 = load i32, ptr %137, align 4
  %139 = sext i32 %138 to i64
  %140 = add nsw i64 0, %139
  %141 = getelementptr inbounds i8, ptr %134, i64 %140
  %142 = call noundef ptr @"?rdbuf@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_streambuf@DU?$char_traits@D@std@@@2@XZ"(ptr noundef nonnull align 8 dereferenceable(96) %141) #3
  %143 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %144 = getelementptr inbounds i8, ptr %143, i64 0
  %145 = load ptr, ptr %144, align 8
  %146 = getelementptr inbounds i32, ptr %145, i32 1
  %147 = load i32, ptr %146, align 4
  %148 = sext i32 %147 to i64
  %149 = add nsw i64 0, %148
  %150 = getelementptr inbounds i8, ptr %143, i64 %149
  %151 = call noundef i8 @"?fill@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADXZ"(ptr noundef nonnull align 8 dereferenceable(96) %150) #3
  %152 = invoke noundef i32 @"?sputc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHD@Z"(ptr noundef nonnull align 8 dereferenceable(104) %142, i8 noundef %151)
          to label %169 unwind label %153

153:                                              ; preds = %133, %107, %73, %56
  %154 = catchswitch within none [label %155] unwind label %203

155:                                              ; preds = %153
  %156 = catchpad within %154 [ptr null, i32 0, ptr null]
  %157 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %158 = getelementptr inbounds i8, ptr %157, i64 0
  %159 = load ptr, ptr %158, align 8
  %160 = getelementptr inbounds i32, ptr %159, i32 1
  %161 = load i32, ptr %160, align 4
  %162 = sext i32 %161 to i64
  %163 = add nsw i64 0, %162
  %164 = getelementptr inbounds i8, ptr %157, i64 %163
  invoke void @"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"(ptr noundef nonnull align 8 dereferenceable(96) %164, i32 noundef 4, i1 noundef zeroext true) [ "funclet"(token %156) ]
          to label %165 unwind label %203

165:                                              ; preds = %155
  invoke void @llvm.seh.scope.end() [ "funclet"(token %156) ]
          to label %166 unwind label %203

166:                                              ; preds = %165
  catchret from %156 to label %167

167:                                              ; preds = %166
  br label %168

168:                                              ; preds = %167, %180
  br label %190

169:                                              ; preds = %133
  %170 = call noundef i32 @"?eof@?$_Narrow_char_traits@DH@std@@SAHXZ"() #3
  %171 = call noundef zeroext i1 @"?eq_int_type@?$_Narrow_char_traits@DH@std@@SA_NHH@Z"(i32 noundef %170, i32 noundef %152) #3
  br i1 %171, label %172, label %175

172:                                              ; preds = %169
  %173 = load i32, ptr %5, align 4
  %174 = or i32 %173, 4
  store i32 %174, ptr %5, align 4
  br label %179

175:                                              ; preds = %169
  br label %176

176:                                              ; preds = %175
  %177 = load i64, ptr %7, align 8
  %178 = add nsw i64 %177, -1
  store i64 %178, ptr %7, align 8
  br label %130, !llvm.loop !22

179:                                              ; preds = %172, %130
  br label %180

180:                                              ; preds = %179, %126
  %181 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %182 = getelementptr inbounds i8, ptr %181, i64 0
  %183 = load ptr, ptr %182, align 8
  %184 = getelementptr inbounds i32, ptr %183, i32 1
  %185 = load i32, ptr %184, align 4
  %186 = sext i32 %185 to i64
  %187 = add nsw i64 0, %186
  %188 = getelementptr inbounds i8, ptr %181, i64 %187
  %189 = call noundef i64 @"?width@ios_base@std@@QEAA_J_J@Z"(ptr noundef nonnull align 8 dereferenceable(72) %188, i64 noundef 0) #3
  br label %168

190:                                              ; preds = %168, %53
  %191 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %192 = getelementptr inbounds i8, ptr %191, i64 0
  %193 = load ptr, ptr %192, align 8
  %194 = getelementptr inbounds i32, ptr %193, i32 1
  %195 = load i32, ptr %194, align 4
  %196 = sext i32 %195 to i64
  %197 = add nsw i64 0, %196
  %198 = getelementptr inbounds i8, ptr %191, i64 %197
  %199 = load i32, ptr %5, align 4
  invoke void @"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"(ptr noundef nonnull align 8 dereferenceable(96) %198, i32 noundef %199, i1 noundef zeroext false)
          to label %200 unwind label %203

200:                                              ; preds = %190
  %201 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  invoke void @llvm.seh.scope.end()
          to label %202 unwind label %203

202:                                              ; preds = %200
  call void @"??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %8) #3
  ret ptr %201

203:                                              ; preds = %200, %190, %165, %155, %153, %50, %46
  %204 = cleanuppad within none []
  call void @"??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %8) #3 [ "funclet"(token %204) ]
  cleanupret from %204 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?get_id@this_thread@std@@YA?AVid@thread@2@XZ"(ptr dead_on_unwind noalias writable sret(%"class.std::thread::id") align 4 %0) #0 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = call i32 @_Thrd_id()
  %4 = call noundef ptr @"??0id@thread@std@@AEAA@I@Z"(ptr noundef nonnull align 4 dereferenceable(4) %0, i32 noundef %3) #3
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define internal noundef zeroext i1 @"?await_ready@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAA_NXZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) #0 align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret i1 false
}

; Function Attrs: alwaysinline mustprogress
define private void @"?resuming_on_new_thread@@YA?AUtask@@AEAVjthread@std@@@Z.__await_suspend_wrapper__await"(ptr noundef nonnull %0, ptr noundef %1) #8 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca %"struct.std::coroutine_handle", align 8
  %6 = alloca %"struct.std::coroutine_handle.0", align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %7 = load ptr, ptr %4, align 8
  %8 = load ptr, ptr %3, align 8
  call void @"?from_address@?$coroutine_handle@Upromise_type@task@@@std@@SA?AU12@QEAX@Z"(ptr dead_on_unwind writable sret(%"struct.std::coroutine_handle.0") align 8 %6, ptr noundef %8) #3
  call void @"??B?$coroutine_handle@Upromise_type@task@@@std@@QEBA?AU?$coroutine_handle@X@1@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %6, ptr dead_on_unwind writable sret(%"struct.std::coroutine_handle") align 8 %5) #3
  %9 = getelementptr inbounds nuw %"struct.std::coroutine_handle", ptr %5, i32 0, i32 0
  %10 = load ptr, ptr %9, align 8
  %11 = ptrtoint ptr %10 to i64
  call void @"?await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@4@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %7, i64 %11)
  ret void
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define internal void @"?await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@4@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0, i64 %1) #10 align 2 personality ptr @__CxxFrameHandler3 {
  %3 = alloca %"struct.std::coroutine_handle", align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca %"class.std::runtime_error", align 8
  %7 = alloca %"class.std::jthread", align 8
  %8 = alloca %class.anon, align 8
  %9 = alloca %"class.std::thread::id", align 4
  %10 = getelementptr inbounds nuw %"struct.std::coroutine_handle", ptr %3, i32 0, i32 0
  %11 = inttoptr i64 %1 to ptr
  store ptr %11, ptr %10, align 8
  store ptr %0, ptr %4, align 8
  %12 = load ptr, ptr %4, align 8
  %13 = getelementptr inbounds nuw %struct.awaitable, ptr %12, i32 0, i32 0
  %14 = load ptr, ptr %13, align 8
  store ptr %14, ptr %5, align 8
  %15 = load ptr, ptr %5, align 8, !nonnull !16, !align !17
  %16 = call noundef zeroext i1 @"?joinable@jthread@std@@QEBA_NXZ"(ptr noundef nonnull align 8 dereferenceable(24) %15) #3
  br i1 %16, label %17, label %19

17:                                               ; preds = %2
  %18 = call noundef ptr @"??0runtime_error@std@@QEAA@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(24) %6, ptr noundef @"??_C@_0CD@HNLLMDJL@Output?5jthread?5parameter?5not?5emp@")
  call void @_CxxThrowException(ptr %6, ptr @"_TI2?AVruntime_error@std@@") #22
  unreachable

19:                                               ; preds = %2
  %20 = getelementptr inbounds nuw %class.anon, ptr %8, i32 0, i32 0
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %20, ptr align 8 %3, i64 8, i1 false)
  %21 = call noundef ptr @"??$?0V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@5@@Z@$$V$0A@@jthread@std@@QEAA@$$QEAV<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAV01@@Z@QEAAXU?$coroutine_handle@X@1@@Z@@Z"(ptr noundef nonnull align 8 dereferenceable(24) %7, ptr noundef nonnull align 8 dereferenceable(8) %8)
  invoke void @llvm.seh.scope.begin()
          to label %22 unwind label %32

22:                                               ; preds = %19
  %23 = load ptr, ptr %5, align 8, !nonnull !16, !align !17
  %24 = call noundef nonnull align 8 dereferenceable(24) ptr @"??4jthread@std@@QEAAAEAV01@$$QEAV01@@Z"(ptr noundef nonnull align 8 dereferenceable(24) %23, ptr noundef nonnull align 8 dereferenceable(24) %7) #3
  invoke void @llvm.seh.scope.end()
          to label %25 unwind label %32

25:                                               ; preds = %22
  call void @"??1jthread@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %7) #3
  %26 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(8) @"?cout@std@@3V?$basic_ostream@DU?$char_traits@D@std@@@1@A", ptr noundef @"??_C@_0BA@OADHDNAI@New?5thread?5ID?3?5?$AA@")
  %27 = load ptr, ptr %5, align 8, !nonnull !16, !align !17
  call void @"?get_id@jthread@std@@QEBA?AVid@thread@2@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %27, ptr dead_on_unwind writable sret(%"class.std::thread::id") align 4 %9) #3
  %28 = getelementptr inbounds nuw %"class.std::thread::id", ptr %9, i32 0, i32 0
  %29 = load i32, ptr %28, align 4
  %30 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$?6DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@Vid@thread@0@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %26, i32 %29)
  %31 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@D@Z"(ptr noundef nonnull align 8 dereferenceable(8) %30, i8 noundef 10)
  ret void

32:                                               ; preds = %22, %19
  %33 = cleanuppad within none []
  call void @"??1jthread@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %7) #3 [ "funclet"(token %33) ]
  cleanupret from %33 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define internal void @"?await_resume@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXXZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) #0 align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?return_void@promise_type@task@@QEAAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?unhandled_exception@promise_type@task@@QEAAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret void
}

; Function Attrs: nounwind memory(none)
declare dso_local void @llvm.seh.scope.end() #5

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?final_suspend@promise_type@task@@QEAA?AUsuspend_never@std@@XZ"(ptr noundef nonnull align 1 dereferenceable(1) %0, ptr dead_on_unwind noalias writable sret(%"struct.std::suspend_never") align 1 %1) #0 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  ret void
}

; Function Attrs: alwaysinline mustprogress
define private void @"?resuming_on_new_thread@@YA?AUtask@@AEAVjthread@std@@@Z.__await_suspend_wrapper__final"(ptr noundef nonnull %0, ptr noundef %1) #8 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca %"struct.std::coroutine_handle", align 8
  %6 = alloca %"struct.std::coroutine_handle.0", align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %7 = load ptr, ptr %4, align 8
  %8 = load ptr, ptr %3, align 8
  call void @"?from_address@?$coroutine_handle@Upromise_type@task@@@std@@SA?AU12@QEAX@Z"(ptr dead_on_unwind writable sret(%"struct.std::coroutine_handle.0") align 8 %6, ptr noundef %8) #3
  call void @"??B?$coroutine_handle@Upromise_type@task@@@std@@QEBA?AU?$coroutine_handle@X@1@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %6, ptr dead_on_unwind writable sret(%"struct.std::coroutine_handle") align 8 %5) #3
  %9 = getelementptr inbounds nuw %"struct.std::coroutine_handle", ptr %5, i32 0, i32 0
  %10 = load ptr, ptr %9, align 8
  %11 = ptrtoint ptr %10 to i64
  call void @"?await_suspend@suspend_never@std@@QEBAXU?$coroutine_handle@X@2@@Z"(ptr noundef nonnull align 1 dereferenceable(1) %7, i64 %11) #3
  ret void
}

; Function Attrs: nounwind
declare i1 @llvm.coro.end(ptr, i1, token) #3

; Function Attrs: nobuiltin nounwind
declare dso_local void @"??3@YAXPEAX_K@Z"(ptr noundef, i64 noundef) #11

; Function Attrs: nounwind memory(argmem: read)
declare ptr @llvm.coro.free(token, ptr readonly captures(none)) #12

; Function Attrs: mustprogress noinline norecurse optnone sspstrong uwtable
define dso_local noundef i32 @main() #13 personality ptr @__CxxFrameHandler3 {
  %1 = alloca %"class.std::jthread", align 8
  %2 = alloca %struct.task, align 1
  %3 = call noundef ptr @"??0jthread@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %1) #3
  invoke void @llvm.seh.scope.begin()
          to label %4 unwind label %8

4:                                                ; preds = %0
  %5 = invoke i8 @"?resuming_on_new_thread@@YA?AUtask@@AEAVjthread@std@@@Z"(ptr noundef nonnull align 8 dereferenceable(24) %1)
          to label %6 unwind label %8

6:                                                ; preds = %4
  invoke void @llvm.seh.scope.end()
          to label %7 unwind label %8

7:                                                ; preds = %6
  call void @"??1jthread@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %1) #3
  ret i32 0

8:                                                ; preds = %6, %4, %0
  %9 = cleanuppad within none []
  call void @"??1jthread@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %1) #3 [ "funclet"(token %9) ]
  cleanupret from %9 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0jthread@std@@QEAA@XZ"(ptr noundef nonnull returned align 8 dereferenceable(24) %0) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %2 = alloca ptr, align 8
  %3 = alloca %"struct.std::nostopstate_t", align 1
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr inbounds nuw %"class.std::jthread", ptr %4, i32 0, i32 0
  %6 = call noundef ptr @"??0thread@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %5) #3
  invoke void @llvm.seh.scope.begin()
          to label %7 unwind label %18

7:                                                ; preds = %1
  %8 = getelementptr inbounds nuw %"class.std::jthread", ptr %4, i32 0, i32 1
  %9 = getelementptr inbounds nuw %"struct.std::nostopstate_t", ptr %3, i32 0, i32 0
  %10 = load i8, ptr %9, align 1
  %11 = call noundef ptr @"??0stop_source@std@@QEAA@Unostopstate_t@1@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %8, i8 %10) #3
  invoke void @llvm.seh.scope.begin()
          to label %12 unwind label %15

12:                                               ; preds = %7
  invoke void @llvm.seh.scope.end()
          to label %13 unwind label %15

13:                                               ; preds = %12
  invoke void @llvm.seh.scope.end()
          to label %14 unwind label %18

14:                                               ; preds = %13
  ret ptr %4

15:                                               ; preds = %12, %7
  %16 = cleanuppad within none []
  invoke void @llvm.seh.scope.end() [ "funclet"(token %16) ]
          to label %17 unwind label %18

17:                                               ; preds = %15
  call void @"??1stop_source@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %8) #3 [ "funclet"(token %16) ]
  cleanupret from %16 unwind label %18

18:                                               ; preds = %13, %17, %15, %1
  %19 = cleanuppad within none []
  call void @"??1thread@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %5) #3 [ "funclet"(token %19) ]
  cleanupret from %19 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"??1jthread@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  invoke void @llvm.seh.scope.begin()
          to label %4 unwind label %14

4:                                                ; preds = %1
  invoke void @llvm.seh.scope.begin()
          to label %5 unwind label %10

5:                                                ; preds = %4
  call void @"?_Try_cancel_and_join@jthread@std@@AEAAXXZ"(ptr noundef nonnull align 8 dereferenceable(24) %3) #3
  invoke void @llvm.seh.scope.end()
          to label %6 unwind label %10

6:                                                ; preds = %5
  %7 = getelementptr inbounds nuw %"class.std::jthread", ptr %3, i32 0, i32 1
  call void @"??1stop_source@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %7) #3
  invoke void @llvm.seh.scope.end()
          to label %8 unwind label %14

8:                                                ; preds = %6
  %9 = getelementptr inbounds nuw %"class.std::jthread", ptr %3, i32 0, i32 0
  call void @"??1thread@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %9) #3
  ret void

10:                                               ; preds = %5, %4
  %11 = cleanuppad within none []
  invoke void @llvm.seh.scope.end() [ "funclet"(token %11) ]
          to label %12 unwind label %14

12:                                               ; preds = %10
  %13 = getelementptr inbounds nuw %"class.std::jthread", ptr %3, i32 0, i32 1
  call void @"??1stop_source@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %13) #3 [ "funclet"(token %11) ]
  cleanupret from %11 unwind label %14

14:                                               ; preds = %6, %12, %10, %1
  %15 = cleanuppad within none []
  %16 = getelementptr inbounds nuw %"class.std::jthread", ptr %3, i32 0, i32 0
  call void @"??1thread@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %16) #3 [ "funclet"(token %15) ]
  cleanupret from %15 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0?$coroutine_handle@Upromise_type@task@@@std@@QEAA@XZ"(ptr noundef nonnull returned align 8 dereferenceable(8) %0) unnamed_addr #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"struct.std::coroutine_handle.0", ptr %3, i32 0, i32 0
  store ptr null, ptr %4, align 8
  ret ptr %3
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?from_address@?$coroutine_handle@X@std@@SA?AU12@QEAX@Z"(ptr dead_on_unwind noalias writable sret(%"struct.std::coroutine_handle") align 8 %0, ptr noundef %1) #0 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %5 = call noundef ptr @"??0?$coroutine_handle@X@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) #3
  %6 = load ptr, ptr %4, align 8
  %7 = getelementptr inbounds nuw %"struct.std::coroutine_handle", ptr %0, i32 0, i32 0
  store ptr %6, ptr %7, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0?$coroutine_handle@X@std@@QEAA@XZ"(ptr noundef nonnull returned align 8 dereferenceable(8) %0) unnamed_addr #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"struct.std::coroutine_handle", ptr %3, i32 0, i32 0
  store ptr null, ptr %4, align 8
  ret ptr %3
}

declare dso_local i32 @_Thrd_id() #14

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0id@thread@std@@AEAA@I@Z"(ptr noundef nonnull returned align 4 dereferenceable(4) %0, i32 noundef %1) unnamed_addr #0 comdat align 2 {
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  store i32 %1, ptr %3, align 4
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr inbounds nuw %"class.std::thread::id", ptr %5, i32 0, i32 0
  %7 = load i32, ptr %3, align 4
  store i32 %7, ptr %6, align 4
  ret ptr %5
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef zeroext i1 @"?joinable@jthread@std@@QEBA_NXZ"(ptr noundef nonnull align 8 dereferenceable(24) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"class.std::jthread", ptr %3, i32 0, i32 0
  %5 = call noundef zeroext i1 @"?joinable@thread@std@@QEBA_NXZ"(ptr noundef nonnull align 8 dereferenceable(16) %4) #3
  ret i1 %5
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0runtime_error@std@@QEAA@PEBD@Z"(ptr noundef nonnull returned align 8 dereferenceable(24) %0, ptr noundef %1) unnamed_addr #10 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = load ptr, ptr %3, align 8
  %7 = call noundef ptr @"??0exception@std@@QEAA@QEBD@Z"(ptr noundef nonnull align 8 dereferenceable(24) %5, ptr noundef %6) #3
  invoke void @llvm.seh.scope.begin()
          to label %8 unwind label %10

8:                                                ; preds = %2
  store ptr @"??_7runtime_error@std@@6B@", ptr %5, align 8
  invoke void @llvm.seh.scope.end()
          to label %9 unwind label %10

9:                                                ; preds = %8
  ret ptr %5

10:                                               ; preds = %8, %2
  %11 = cleanuppad within none []
  call void @"??1exception@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %5) #3 [ "funclet"(token %11) ]
  cleanupret from %11 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0runtime_error@std@@QEAA@AEBV01@@Z"(ptr noundef nonnull returned align 8 dereferenceable(24) %0, ptr noundef nonnull align 8 dereferenceable(24) %1) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %7 = call noundef ptr @"??0exception@std@@QEAA@AEBV01@@Z"(ptr noundef nonnull align 8 dereferenceable(24) %5, ptr noundef nonnull align 8 dereferenceable(24) %6) #3
  invoke void @llvm.seh.scope.begin()
          to label %8 unwind label %10

8:                                                ; preds = %2
  store ptr @"??_7runtime_error@std@@6B@", ptr %5, align 8
  invoke void @llvm.seh.scope.end()
          to label %9 unwind label %10

9:                                                ; preds = %8
  ret ptr %5

10:                                               ; preds = %8, %2
  %11 = cleanuppad within none []
  call void @"??1exception@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %5) #3 [ "funclet"(token %11) ]
  cleanupret from %11 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0exception@std@@QEAA@AEBV01@@Z"(ptr noundef nonnull returned align 8 dereferenceable(24) %0, ptr noundef nonnull align 8 dereferenceable(24) %1) unnamed_addr #0 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  store ptr @"??_7exception@std@@6B@", ptr %5, align 8
  %6 = getelementptr inbounds nuw %"class.std::exception", ptr %5, i32 0, i32 1
  call void @llvm.memset.p0.i64(ptr align 8 %6, i8 0, i64 16, i1 false)
  %7 = getelementptr inbounds nuw %"class.std::exception", ptr %5, i32 0, i32 1
  %8 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %9 = getelementptr inbounds nuw %"class.std::exception", ptr %8, i32 0, i32 1
  call void @__std_exception_copy(ptr noundef %9, ptr noundef %7)
  ret ptr %5
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"??1runtime_error@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  invoke void @llvm.seh.scope.begin()
          to label %4 unwind label %6

4:                                                ; preds = %1
  invoke void @llvm.seh.scope.end()
          to label %5 unwind label %6

5:                                                ; preds = %4
  call void @"??1exception@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %3) #3
  ret void

6:                                                ; preds = %4, %1
  %7 = cleanuppad within none []
  call void @"??1exception@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %3) #3 [ "funclet"(token %7) ]
  cleanupret from %7 unwind to caller
}

declare dso_local void @_CxxThrowException(ptr, ptr)

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #15

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define internal noundef ptr @"??$?0V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@5@@Z@$$V$0A@@jthread@std@@QEAA@$$QEAV<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAV01@@Z@QEAAXU?$coroutine_handle@X@1@@Z@@Z"(ptr noundef nonnull returned align 8 dereferenceable(24) %0, ptr noundef nonnull align 8 dereferenceable(8) %1) unnamed_addr #10 align 2 personality ptr @__CxxFrameHandler3 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr inbounds nuw %"class.std::jthread", ptr %5, i32 0, i32 0
  %7 = call noundef ptr @"??0thread@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %6) #3
  invoke void @llvm.seh.scope.begin()
          to label %8 unwind label %21

8:                                                ; preds = %2
  %9 = getelementptr inbounds nuw %"class.std::jthread", ptr %5, i32 0, i32 1
  %10 = invoke noundef ptr @"??0stop_source@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %9)
          to label %11 unwind label %21

11:                                               ; preds = %8
  invoke void @llvm.seh.scope.begin()
          to label %12 unwind label %18

12:                                               ; preds = %11
  %13 = getelementptr inbounds nuw %"class.std::jthread", ptr %5, i32 0, i32 0
  %14 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  invoke void @"??$_Start@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@$$V@thread@std@@AEAAX$$QEAV<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@1@@Z@QEAAXU?$coroutine_handle@X@1@@Z@@Z"(ptr noundef nonnull align 8 dereferenceable(16) %13, ptr noundef nonnull align 8 dereferenceable(8) %14)
          to label %15 unwind label %18

15:                                               ; preds = %12
  invoke void @llvm.seh.scope.end()
          to label %16 unwind label %18

16:                                               ; preds = %15
  invoke void @llvm.seh.scope.end()
          to label %17 unwind label %21

17:                                               ; preds = %16
  ret ptr %5

18:                                               ; preds = %15, %12, %11
  %19 = cleanuppad within none []
  invoke void @llvm.seh.scope.end() [ "funclet"(token %19) ]
          to label %20 unwind label %21

20:                                               ; preds = %18
  call void @"??1stop_source@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %9) #3 [ "funclet"(token %19) ]
  cleanupret from %19 unwind label %21

21:                                               ; preds = %16, %20, %18, %8, %2
  %22 = cleanuppad within none []
  call void @"??1thread@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %6) #3 [ "funclet"(token %22) ]
  cleanupret from %22 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(24) ptr @"??4jthread@std@@QEAAAEAV01@$$QEAV01@@Z"(ptr noundef nonnull align 8 dereferenceable(24) %0, ptr noundef nonnull align 8 dereferenceable(24) %1) #0 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  store ptr %1, ptr %4, align 8
  store ptr %0, ptr %5, align 8
  %6 = load ptr, ptr %5, align 8
  %7 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %8 = icmp eq ptr %6, %7
  br i1 %8, label %9, label %10

9:                                                ; preds = %2
  store ptr %6, ptr %3, align 8
  br label %19

10:                                               ; preds = %2
  call void @"?_Try_cancel_and_join@jthread@std@@AEAAXXZ"(ptr noundef nonnull align 8 dereferenceable(24) %6) #3
  %11 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %12 = getelementptr inbounds nuw %"class.std::jthread", ptr %11, i32 0, i32 0
  %13 = getelementptr inbounds nuw %"class.std::jthread", ptr %6, i32 0, i32 0
  %14 = call noundef nonnull align 8 dereferenceable(16) ptr @"??4thread@std@@QEAAAEAV01@$$QEAV01@@Z"(ptr noundef nonnull align 8 dereferenceable(16) %13, ptr noundef nonnull align 8 dereferenceable(16) %12) #3
  %15 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %16 = getelementptr inbounds nuw %"class.std::jthread", ptr %15, i32 0, i32 1
  %17 = getelementptr inbounds nuw %"class.std::jthread", ptr %6, i32 0, i32 1
  %18 = call noundef nonnull align 8 dereferenceable(8) ptr @"??4stop_source@std@@QEAAAEAV01@$$QEAV01@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %17, ptr noundef nonnull align 8 dereferenceable(8) %16) #3
  store ptr %6, ptr %3, align 8
  br label %19

19:                                               ; preds = %10, %9
  %20 = load ptr, ptr %3, align 8
  ret ptr %20
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?get_id@jthread@std@@QEBA?AVid@thread@2@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %0, ptr dead_on_unwind noalias writable sret(%"class.std::thread::id") align 4 %1) #0 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr inbounds nuw %"class.std::jthread", ptr %5, i32 0, i32 0
  call void @"?get_id@thread@std@@QEBA?AVid@12@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %6, ptr dead_on_unwind writable sret(%"class.std::thread::id") align 4 %1) #3
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef zeroext i1 @"?joinable@thread@std@@QEBA_NXZ"(ptr noundef nonnull align 8 dereferenceable(16) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"class.std::thread", ptr %3, i32 0, i32 0
  %5 = getelementptr inbounds nuw %struct._Thrd_t, ptr %4, i32 0, i32 1
  %6 = load i32, ptr %5, align 8
  %7 = icmp ne i32 %6, 0
  ret i1 %7
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0exception@std@@QEAA@QEBD@Z"(ptr noundef nonnull returned align 8 dereferenceable(24) %0, ptr noundef %1) unnamed_addr #0 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca %struct.__std_exception_data, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %6 = load ptr, ptr %4, align 8
  store ptr @"??_7exception@std@@6B@", ptr %6, align 8
  %7 = getelementptr inbounds nuw %"class.std::exception", ptr %6, i32 0, i32 1
  call void @llvm.memset.p0.i64(ptr align 8 %7, i8 0, i64 16, i1 false)
  %8 = getelementptr inbounds nuw %struct.__std_exception_data, ptr %5, i32 0, i32 0
  %9 = load ptr, ptr %3, align 8
  store ptr %9, ptr %8, align 8
  %10 = getelementptr inbounds nuw %struct.__std_exception_data, ptr %5, i32 0, i32 1
  store i8 1, ptr %10, align 8
  %11 = getelementptr inbounds nuw %"class.std::exception", ptr %6, i32 0, i32 1
  call void @__std_exception_copy(ptr noundef %5, ptr noundef %11)
  ret ptr %6
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"??1exception@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  store ptr @"??_7exception@std@@6B@", ptr %3, align 8
  %4 = getelementptr inbounds nuw %"class.std::exception", ptr %3, i32 0, i32 1
  call void @__std_exception_destroy(ptr noundef %4)
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??_Gruntime_error@std@@UEAAPEAXI@Z"(ptr noundef nonnull align 8 dereferenceable(24) %0, i32 noundef %1) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  %5 = alloca ptr, align 8
  store i32 %1, ptr %4, align 4
  store ptr %0, ptr %5, align 8
  %6 = load ptr, ptr %5, align 8
  store ptr %6, ptr %3, align 8
  %7 = load i32, ptr %4, align 4
  invoke void @llvm.seh.scope.begin()
          to label %8 unwind label %14

8:                                                ; preds = %2
  call void @"??1runtime_error@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %6) #3
  invoke void @llvm.seh.scope.end()
          to label %9 unwind label %14

9:                                                ; preds = %8
  %10 = icmp eq i32 %7, 0
  br i1 %10, label %12, label %11

11:                                               ; preds = %9
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %6, i64 noundef 24) #23
  br label %12

12:                                               ; preds = %11, %9
  %13 = load ptr, ptr %3, align 8
  ret ptr %13

14:                                               ; preds = %8, %2
  %15 = cleanuppad within none []
  %16 = icmp eq i32 %7, 0
  br i1 %16, label %18, label %17

17:                                               ; preds = %14
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %6, i64 noundef 24) #23 [ "funclet"(token %15) ]
  br label %18

18:                                               ; preds = %17, %14
  cleanupret from %15 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"?what@exception@std@@UEBAPEBDXZ"(ptr noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"class.std::exception", ptr %3, i32 0, i32 1
  %5 = getelementptr inbounds nuw %struct.__std_exception_data, ptr %4, i32 0, i32 0
  %6 = load ptr, ptr %5, align 8
  %7 = icmp ne ptr %6, null
  br i1 %7, label %8, label %12

8:                                                ; preds = %1
  %9 = getelementptr inbounds nuw %"class.std::exception", ptr %3, i32 0, i32 1
  %10 = getelementptr inbounds nuw %struct.__std_exception_data, ptr %9, i32 0, i32 0
  %11 = load ptr, ptr %10, align 8
  br label %13

12:                                               ; preds = %1
  br label %13

13:                                               ; preds = %12, %8
  %14 = phi ptr [ %11, %8 ], [ @"??_C@_0BC@EOODALEL@Unknown?5exception?$AA@", %12 ]
  ret ptr %14
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #16

declare dso_local void @__std_exception_copy(ptr noundef, ptr noundef) #14

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??_Gexception@std@@UEAAPEAXI@Z"(ptr noundef nonnull align 8 dereferenceable(24) %0, i32 noundef %1) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  %5 = alloca ptr, align 8
  store i32 %1, ptr %4, align 4
  store ptr %0, ptr %5, align 8
  %6 = load ptr, ptr %5, align 8
  store ptr %6, ptr %3, align 8
  %7 = load i32, ptr %4, align 4
  invoke void @llvm.seh.scope.begin()
          to label %8 unwind label %14

8:                                                ; preds = %2
  call void @"??1exception@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %6) #3
  invoke void @llvm.seh.scope.end()
          to label %9 unwind label %14

9:                                                ; preds = %8
  %10 = icmp eq i32 %7, 0
  br i1 %10, label %12, label %11

11:                                               ; preds = %9
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %6, i64 noundef 24) #23
  br label %12

12:                                               ; preds = %11, %9
  %13 = load ptr, ptr %3, align 8
  ret ptr %13

14:                                               ; preds = %8, %2
  %15 = cleanuppad within none []
  %16 = icmp eq i32 %7, 0
  br i1 %16, label %18, label %17

17:                                               ; preds = %14
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %6, i64 noundef 24) #23 [ "funclet"(token %15) ]
  br label %18

18:                                               ; preds = %17, %14
  cleanupret from %15 unwind to caller
}

declare dso_local void @__std_exception_destroy(ptr noundef) #14

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0thread@std@@QEAA@XZ"(ptr noundef nonnull returned align 8 dereferenceable(16) %0) unnamed_addr #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"class.std::thread", ptr %3, i32 0, i32 0
  %5 = getelementptr inbounds nuw %struct._Thrd_t, ptr %4, i32 0, i32 0
  store ptr null, ptr %5, align 8
  %6 = getelementptr inbounds nuw %struct._Thrd_t, ptr %4, i32 0, i32 1
  store i32 0, ptr %6, align 8
  ret ptr %3
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0stop_source@std@@QEAA@XZ"(ptr noundef nonnull returned align 8 dereferenceable(8) %0) unnamed_addr #10 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"class.std::stop_source", ptr %3, i32 0, i32 0
  %5 = call noalias noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef 32) #24
  invoke void @llvm.seh.scope.begin()
          to label %6 unwind label %9

6:                                                ; preds = %1
  %7 = call noundef ptr @"??0_Stop_state@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %5) #3
  invoke void @llvm.seh.scope.end()
          to label %8 unwind label %9

8:                                                ; preds = %6
  store ptr %5, ptr %4, align 8
  ret ptr %3

9:                                                ; preds = %6, %1
  %10 = cleanuppad within none []
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %5, i64 noundef 32) #23 [ "funclet"(token %10) ]
  cleanupret from %10 unwind to caller
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define internal void @"??$_Start@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@$$V@thread@std@@AEAAX$$QEAV<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@1@@Z@QEAAXU?$coroutine_handle@X@1@@Z@@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(8) %1) #10 align 2 personality ptr @__CxxFrameHandler3 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca %"class.std::unique_ptr", align 8
  %6 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %7 = load ptr, ptr %4, align 8
  %8 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  call void @"??$make_unique@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@2@@Z@QEAAXU?$coroutine_handle@X@2@@Z@$0A@@std@@YA?AV?$unique_ptr@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@U?$default_delete@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@@2@@0@$$QEAV<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@0@@Z@QEAAXU?$coroutine_handle@X@0@@Z@@Z"(ptr dead_on_unwind writable sret(%"class.std::unique_ptr") align 8 %5, ptr noundef nonnull align 8 dereferenceable(8) %8)
  invoke void @llvm.seh.scope.begin()
          to label %9 unwind label %30

9:                                                ; preds = %2
  store ptr @"??$_Invoke@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@$0A@@thread@std@@CAIPEAX@Z", ptr %6, align 8
  %10 = getelementptr inbounds nuw %"class.std::thread", ptr %7, i32 0, i32 0
  %11 = getelementptr inbounds nuw %struct._Thrd_t, ptr %10, i32 0, i32 1
  %12 = call noundef ptr @"?get@?$unique_ptr@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@U?$default_delete@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@@2@@std@@QEBAPEAV?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@2@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %5) #3
  %13 = invoke i64 @_beginthreadex(ptr noundef null, i32 noundef 0, ptr noundef @"??$_Invoke@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@$0A@@thread@std@@CAIPEAX@Z", ptr noundef %12, i32 noundef 0, ptr noundef %11)
          to label %14 unwind label %30

14:                                               ; preds = %9
  %15 = inttoptr i64 %13 to ptr
  %16 = getelementptr inbounds nuw %"class.std::thread", ptr %7, i32 0, i32 0
  %17 = getelementptr inbounds nuw %struct._Thrd_t, ptr %16, i32 0, i32 0
  store ptr %15, ptr %17, align 8
  %18 = getelementptr inbounds nuw %"class.std::thread", ptr %7, i32 0, i32 0
  %19 = getelementptr inbounds nuw %struct._Thrd_t, ptr %18, i32 0, i32 0
  %20 = load ptr, ptr %19, align 8
  %21 = icmp ne ptr %20, null
  br i1 %21, label %22, label %24

22:                                               ; preds = %14
  %23 = call noundef ptr @"?release@?$unique_ptr@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@U?$default_delete@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@@2@@std@@QEAAPEAV?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@2@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %5) #3
  br label %28

24:                                               ; preds = %14
  %25 = getelementptr inbounds nuw %"class.std::thread", ptr %7, i32 0, i32 0
  %26 = getelementptr inbounds nuw %struct._Thrd_t, ptr %25, i32 0, i32 1
  store i32 0, ptr %26, align 8
  invoke void @"?_Throw_Cpp_error@std@@YAXH@Z"(i32 noundef 6) #22
          to label %27 unwind label %30

27:                                               ; preds = %24
  unreachable

28:                                               ; preds = %22
  invoke void @llvm.seh.scope.end()
          to label %29 unwind label %30

29:                                               ; preds = %28
  call void @"??1?$unique_ptr@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@U?$default_delete@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@@2@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %5) #3
  ret void

30:                                               ; preds = %28, %24, %9, %2
  %31 = cleanuppad within none []
  call void @"??1?$unique_ptr@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@U?$default_delete@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@@2@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %5) #3 [ "funclet"(token %31) ]
  cleanupret from %31 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"??1stop_source@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr inbounds nuw %"class.std::stop_source", ptr %4, i32 0, i32 0
  %6 = load ptr, ptr %5, align 8
  store ptr %6, ptr %3, align 8
  %7 = load ptr, ptr %3, align 8
  %8 = icmp ne ptr %7, null
  br i1 %8, label %9, label %31

9:                                                ; preds = %1
  %10 = load ptr, ptr %3, align 8
  %11 = getelementptr inbounds nuw %"struct.std::_Stop_state", ptr %10, i32 0, i32 1
  %12 = call noundef i32 @"?fetch_sub@?$_Atomic_integral_facade@I@std@@QEAAIIW4memory_order@2@@Z"(ptr noundef nonnull align 4 dereferenceable(4) %11, i32 noundef 2, i32 noundef 4) #3
  %13 = lshr i32 %12, 1
  %14 = icmp eq i32 %13, 1
  br i1 %14, label %15, label %30

15:                                               ; preds = %9
  %16 = load ptr, ptr %3, align 8
  %17 = getelementptr inbounds nuw %"struct.std::_Stop_state", ptr %16, i32 0, i32 0
  %18 = call noundef i32 @"?fetch_sub@?$_Atomic_integral_facade@I@std@@QEAAIIW4memory_order@2@@Z"(ptr noundef nonnull align 4 dereferenceable(4) %17, i32 noundef 1, i32 noundef 4) #3
  %19 = icmp eq i32 %18, 1
  br i1 %19, label %20, label %29

20:                                               ; preds = %15
  %21 = load ptr, ptr %3, align 8
  %22 = icmp eq ptr %21, null
  br i1 %22, label %26, label %23

23:                                               ; preds = %20
  invoke void @llvm.seh.scope.begin()
          to label %24 unwind label %27

24:                                               ; preds = %23
  invoke void @llvm.seh.scope.end()
          to label %25 unwind label %27

25:                                               ; preds = %24
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %21, i64 noundef 32) #23
  br label %26

26:                                               ; preds = %25, %20
  br label %29

27:                                               ; preds = %24, %23
  %28 = cleanuppad within none []
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %21, i64 noundef 32) #23 [ "funclet"(token %28) ]
  cleanupret from %28 unwind to caller

29:                                               ; preds = %26, %15
  br label %30

30:                                               ; preds = %29, %9
  br label %31

31:                                               ; preds = %30, %1
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"??1thread@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %0) unnamed_addr #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call noundef zeroext i1 @"?joinable@thread@std@@QEBA_NXZ"(ptr noundef nonnull align 8 dereferenceable(16) %3) #3
  br i1 %4, label %5, label %6

5:                                                ; preds = %1
  call void @terminate() #25
  unreachable

6:                                                ; preds = %1
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0_Stop_state@std@@QEAA@XZ"(ptr noundef nonnull returned align 8 dereferenceable(32) %0) unnamed_addr #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"struct.std::_Stop_state", ptr %3, i32 0, i32 0
  %5 = call noundef ptr @"??0?$atomic@I@std@@QEAA@I@Z"(ptr noundef nonnull align 4 dereferenceable(4) %4, i32 noundef 1) #3
  %6 = getelementptr inbounds nuw %"struct.std::_Stop_state", ptr %3, i32 0, i32 1
  %7 = call noundef ptr @"??0?$atomic@I@std@@QEAA@I@Z"(ptr noundef nonnull align 4 dereferenceable(4) %6, i32 noundef 2) #3
  %8 = getelementptr inbounds nuw %"struct.std::_Stop_state", ptr %3, i32 0, i32 2
  %9 = call noundef ptr @"??0?$_Locked_pointer@V_Stop_callback_base@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %8) #3
  %10 = getelementptr inbounds nuw %"struct.std::_Stop_state", ptr %3, i32 0, i32 3
  %11 = call noundef ptr @"??0?$atomic@PEBV_Stop_callback_base@std@@@std@@QEAA@QEBV_Stop_callback_base@1@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %10, ptr noundef null) #3
  %12 = getelementptr inbounds nuw %"struct.std::_Stop_state", ptr %3, i32 0, i32 4
  store i32 0, ptr %12, align 8
  ret ptr %3
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define internal noundef ptr @"??0?$atomic@I@std@@QEAA@I@Z"(ptr noundef nonnull returned align 4 dereferenceable(4) %0, i32 noundef %1) unnamed_addr #0 align 2 {
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  store i32 %1, ptr %3, align 4
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = load i32, ptr %3, align 4
  %7 = call noundef ptr @"??0?$_Atomic_integral_facade@I@std@@QEAA@I@Z"(ptr noundef nonnull align 4 dereferenceable(4) %5, i32 noundef %6) #3
  ret ptr %5
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0?$_Locked_pointer@V_Stop_callback_base@std@@@std@@QEAA@XZ"(ptr noundef nonnull returned align 8 dereferenceable(8) %0) unnamed_addr #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"class.std::_Locked_pointer", ptr %3, i32 0, i32 0
  %5 = call noundef ptr @"??0?$atomic@_K@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %4) #3
  ret ptr %3
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define internal noundef ptr @"??0?$atomic@PEBV_Stop_callback_base@std@@@std@@QEAA@QEBV_Stop_callback_base@1@@Z"(ptr noundef nonnull returned align 8 dereferenceable(8) %0, ptr noundef %1) unnamed_addr #0 align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = load ptr, ptr %3, align 8
  %7 = call noundef ptr @"??0?$_Atomic_pointer@PEBV_Stop_callback_base@std@@@std@@QEAA@QEBV_Stop_callback_base@1@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %5, ptr noundef %6) #3
  ret ptr %5
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define internal noundef ptr @"??0?$_Atomic_integral_facade@I@std@@QEAA@I@Z"(ptr noundef nonnull returned align 4 dereferenceable(4) %0, i32 noundef %1) unnamed_addr #0 align 2 {
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  store i32 %1, ptr %3, align 4
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = load i32, ptr %3, align 4
  %7 = call noundef ptr @"??0?$_Atomic_integral@I$03@std@@QEAA@I@Z"(ptr noundef nonnull align 4 dereferenceable(4) %5, i32 noundef %6) #3
  ret ptr %5
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define internal noundef ptr @"??0?$_Atomic_integral@I$03@std@@QEAA@I@Z"(ptr noundef nonnull returned align 4 dereferenceable(4) %0, i32 noundef %1) unnamed_addr #0 align 2 {
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  store i32 %1, ptr %3, align 4
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = load i32, ptr %3, align 4
  %7 = call noundef ptr @"??0?$_Atomic_storage@I$03@std@@QEAA@I@Z"(ptr noundef nonnull align 4 dereferenceable(4) %5, i32 noundef %6) #3
  ret ptr %5
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0?$_Atomic_storage@I$03@std@@QEAA@I@Z"(ptr noundef nonnull returned align 4 dereferenceable(4) %0, i32 noundef %1) unnamed_addr #0 comdat align 2 {
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  store i32 %1, ptr %3, align 4
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr inbounds nuw %"struct.std::_Atomic_storage", ptr %5, i32 0, i32 0
  %7 = getelementptr inbounds nuw %"struct.std::_Atomic_padded", ptr %6, i32 0, i32 0
  %8 = load i32, ptr %3, align 4
  store i32 %8, ptr %7, align 4
  ret ptr %5
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0?$atomic@_K@std@@QEAA@XZ"(ptr noundef nonnull returned align 8 dereferenceable(8) %0) unnamed_addr #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds i8, ptr %3, i64 0
  call void @llvm.memset.p0.i64(ptr align 8 %4, i8 0, i64 8, i1 false)
  ret ptr %3
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define internal noundef ptr @"??0?$_Atomic_pointer@PEBV_Stop_callback_base@std@@@std@@QEAA@QEBV_Stop_callback_base@1@@Z"(ptr noundef nonnull returned align 8 dereferenceable(8) %0, ptr noundef %1) unnamed_addr #0 align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = load ptr, ptr %3, align 8
  %7 = call noundef ptr @"??0?$_Atomic_storage@PEBV_Stop_callback_base@std@@$07@std@@QEAA@QEBV_Stop_callback_base@1@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %5, ptr noundef %6) #3
  ret ptr %5
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0?$_Atomic_storage@PEBV_Stop_callback_base@std@@$07@std@@QEAA@QEBV_Stop_callback_base@1@@Z"(ptr noundef nonnull returned align 8 dereferenceable(8) %0, ptr noundef %1) unnamed_addr #0 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr inbounds nuw %"struct.std::_Atomic_storage.7", ptr %5, i32 0, i32 0
  %7 = getelementptr inbounds nuw %"struct.std::_Atomic_padded.8", ptr %6, i32 0, i32 0
  %8 = load ptr, ptr %3, align 8
  store ptr %8, ptr %7, align 8
  ret ptr %5
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define internal void @"??$make_unique@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@2@@Z@QEAAXU?$coroutine_handle@X@2@@Z@$0A@@std@@YA?AV?$unique_ptr@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@U?$default_delete@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@@2@@0@$$QEAV<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@0@@Z@QEAAXU?$coroutine_handle@X@0@@Z@@Z"(ptr dead_on_unwind noalias writable sret(%"class.std::unique_ptr") align 8 %0, ptr noundef nonnull align 8 dereferenceable(8) %1) #10 personality ptr @__CxxFrameHandler3 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %5 = call noalias noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef 8) #24
  invoke void @llvm.seh.scope.begin()
          to label %6 unwind label %11

6:                                                ; preds = %2
  %7 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %8 = call noundef ptr @"??$?0V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@5@@Z@$$V$0A@@?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@QEAA@$$QEAV<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@1@@Z@QEAAXU?$coroutine_handle@X@1@@Z@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %5, ptr noundef nonnull align 8 dereferenceable(8) %7) #3
  invoke void @llvm.seh.scope.end()
          to label %9 unwind label %11

9:                                                ; preds = %6
  %10 = call noundef ptr @"??$?0U?$default_delete@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@@std@@$0A@@?$unique_ptr@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@U?$default_delete@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@@2@@std@@QEAA@PEAV?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@1@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0, ptr noundef %5) #3
  ret void

11:                                               ; preds = %6, %2
  %12 = cleanuppad within none []
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %5, i64 noundef 8) #23 [ "funclet"(token %12) ]
  cleanupret from %12 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define internal noundef i32 @"??$_Invoke@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@$0A@@thread@std@@CAIPEAX@Z"(ptr noundef %0) #0 align 2 personality ptr @__CxxFrameHandler3 {
  %2 = alloca ptr, align 8
  %3 = alloca %"class.std::unique_ptr", align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %5 = load ptr, ptr %2, align 8
  %6 = call noundef ptr @"??$?0U?$default_delete@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@@std@@$0A@@?$unique_ptr@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@U?$default_delete@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@@2@@std@@QEAA@PEAV?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@1@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %3, ptr noundef %5) #3
  invoke void @llvm.seh.scope.begin()
          to label %7 unwind label %14

7:                                                ; preds = %1
  %8 = call noundef ptr @"?get@?$unique_ptr@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@U?$default_delete@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@@2@@std@@QEBAPEAV?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@2@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %3) #3
  store ptr %8, ptr %4, align 8
  %9 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %10 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$get@$0A@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@YAAEAV<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@0@@Z@QEAAXU?$coroutine_handle@X@0@@Z@AEAV?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@0@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %9) #3
  invoke void @"??$invoke@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@YAX$$QEAV<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@0@@Z@QEAAXU?$coroutine_handle@X@0@@Z@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %10)
          to label %11 unwind label %14

11:                                               ; preds = %7
  invoke void @_Cnd_do_broadcast_at_thread_exit()
          to label %12 unwind label %14

12:                                               ; preds = %11
  invoke void @llvm.seh.scope.end()
          to label %13 unwind label %14

13:                                               ; preds = %12
  call void @"??1?$unique_ptr@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@U?$default_delete@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@@2@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %3) #3
  ret i32 0

14:                                               ; preds = %12, %11, %7, %1
  %15 = cleanuppad within none []
  call void @"??1?$unique_ptr@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@U?$default_delete@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@@2@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %3) #3 [ "funclet"(token %15) ]
  cleanupret from %15 unwind to caller
}

declare dso_local i64 @_beginthreadex(ptr noundef, i32 noundef, ptr noundef, ptr noundef, i32 noundef, ptr noundef) #14

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define internal noundef ptr @"?get@?$unique_ptr@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@U?$default_delete@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@@2@@std@@QEBAPEAV?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@2@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) #0 align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"class.std::unique_ptr", ptr %3, i32 0, i32 0
  %5 = getelementptr inbounds nuw %"class.std::_Compressed_pair", ptr %4, i32 0, i32 0
  %6 = load ptr, ptr %5, align 8
  ret ptr %6
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define internal noundef ptr @"?release@?$unique_ptr@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@U?$default_delete@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@@2@@std@@QEAAPEAV?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@2@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) #0 align 2 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  store ptr null, ptr %3, align 8
  %5 = getelementptr inbounds nuw %"class.std::unique_ptr", ptr %4, i32 0, i32 0
  %6 = getelementptr inbounds nuw %"class.std::_Compressed_pair", ptr %5, i32 0, i32 0
  %7 = call noundef ptr @"??$exchange@PEAV?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@$$T@std@@YAPEAV?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@0@AEAPEAV10@$$QEA$$T@Z"(ptr noundef nonnull align 8 dereferenceable(8) %6, ptr noundef nonnull align 8 dereferenceable(8) %3) #3
  ret ptr %7
}

; Function Attrs: noreturn
declare dso_local void @"?_Throw_Cpp_error@std@@YAXH@Z"(i32 noundef) #17

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define internal void @"??1?$unique_ptr@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@U?$default_delete@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@@2@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) unnamed_addr #0 align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"class.std::unique_ptr", ptr %3, i32 0, i32 0
  %5 = getelementptr inbounds nuw %"class.std::_Compressed_pair", ptr %4, i32 0, i32 0
  %6 = load ptr, ptr %5, align 8
  %7 = icmp ne ptr %6, null
  br i1 %7, label %8, label %14

8:                                                ; preds = %1
  %9 = getelementptr inbounds nuw %"class.std::unique_ptr", ptr %3, i32 0, i32 0
  %10 = call noundef nonnull align 1 dereferenceable(1) ptr @"?_Get_first@?$_Compressed_pair@U?$default_delete@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@@std@@PEAV?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@2@$00@std@@QEAAAEAU?$default_delete@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@@2@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %9) #3
  %11 = getelementptr inbounds nuw %"class.std::unique_ptr", ptr %3, i32 0, i32 0
  %12 = getelementptr inbounds nuw %"class.std::_Compressed_pair", ptr %11, i32 0, i32 0
  %13 = load ptr, ptr %12, align 8
  call void @"??R?$default_delete@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@@std@@QEBAXPEAV?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@1@@Z"(ptr noundef nonnull align 1 dereferenceable(1) %10, ptr noundef %13) #3
  br label %14

14:                                               ; preds = %8, %1
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define internal noundef ptr @"??$?0V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@5@@Z@$$V$0A@@?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@QEAA@$$QEAV<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@1@@Z@QEAAXU?$coroutine_handle@X@1@@Z@@Z"(ptr noundef nonnull returned align 8 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(8) %1) unnamed_addr #0 align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca %"struct.std::_Exact_args_t", align 1
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %8 = getelementptr inbounds nuw %"struct.std::_Exact_args_t", ptr %5, i32 0, i32 0
  %9 = load i8, ptr %8, align 1
  %10 = call noundef ptr @"??$?0U_Exact_args_t@std@@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@1@@Z@QEAAXU?$coroutine_handle@X@1@@Z@$$V$0A@@?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@QEAA@U_Exact_args_t@1@$$QEAV<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@1@@Z@QEAAXU?$coroutine_handle@X@1@@Z@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %6, i8 %9, ptr noundef nonnull align 8 dereferenceable(8) %7)
  ret ptr %6
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define internal noundef ptr @"??$?0U?$default_delete@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@@std@@$0A@@?$unique_ptr@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@U?$default_delete@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@@2@@std@@QEAA@PEAV?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@1@@Z"(ptr noundef nonnull returned align 8 dereferenceable(8) %0, ptr noundef %1) unnamed_addr #0 align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca %"struct.std::_Zero_then_variadic_args_t", align 1
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = getelementptr inbounds nuw %"class.std::unique_ptr", ptr %6, i32 0, i32 0
  %8 = getelementptr inbounds nuw %"struct.std::_Zero_then_variadic_args_t", ptr %5, i32 0, i32 0
  %9 = load i8, ptr %8, align 1
  %10 = call noundef ptr @"??$?0AEAPEAV?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@@?$_Compressed_pair@U?$default_delete@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@@std@@PEAV?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@2@$00@std@@QEAA@U_Zero_then_variadic_args_t@1@AEAPEAV?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@1@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %7, i8 %9, ptr noundef nonnull align 8 dereferenceable(8) %3) #3
  ret ptr %6
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define internal noundef ptr @"??$?0U_Exact_args_t@std@@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@1@@Z@QEAAXU?$coroutine_handle@X@1@@Z@$$V$0A@@?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@QEAA@U_Exact_args_t@1@$$QEAV<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@1@@Z@QEAAXU?$coroutine_handle@X@1@@Z@@Z"(ptr noundef nonnull returned align 8 dereferenceable(8) %0, i8 %1, ptr noundef nonnull align 8 dereferenceable(8) %2) unnamed_addr #10 align 2 {
  %4 = alloca %"struct.std::_Exact_args_t", align 1
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca %"struct.std::_Exact_args_t", align 1
  %8 = getelementptr inbounds nuw %"struct.std::_Exact_args_t", ptr %4, i32 0, i32 0
  store i8 %1, ptr %8, align 1
  store ptr %2, ptr %5, align 8
  store ptr %0, ptr %6, align 8
  %9 = load ptr, ptr %6, align 8
  %10 = getelementptr inbounds nuw %"struct.std::_Exact_args_t", ptr %7, i32 0, i32 0
  %11 = load i8, ptr %10, align 1
  %12 = call noundef ptr @"??$?0U_Exact_args_t@std@@$0A@@?$tuple@$$V@std@@QEAA@U_Exact_args_t@1@@Z"(ptr noundef nonnull align 1 dereferenceable(1) %9, i8 %11) #3
  %13 = getelementptr inbounds nuw %"class.std::tuple", ptr %9, i32 0, i32 0
  %14 = load ptr, ptr %5, align 8, !nonnull !16, !align !17
  %15 = call noundef ptr @"??$?0V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@5@@Z@@?$_Tuple_val@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@QEAA@$$QEAV<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@1@@Z@QEAAXU?$coroutine_handle@X@1@@Z@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %13, ptr noundef nonnull align 8 dereferenceable(8) %14)
  ret ptr %9
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??$?0U_Exact_args_t@std@@$0A@@?$tuple@$$V@std@@QEAA@U_Exact_args_t@1@@Z"(ptr noundef nonnull returned align 1 dereferenceable(1) %0, i8 %1) unnamed_addr #0 comdat align 2 {
  %3 = alloca %"struct.std::_Exact_args_t", align 1
  %4 = alloca ptr, align 8
  %5 = getelementptr inbounds nuw %"struct.std::_Exact_args_t", ptr %3, i32 0, i32 0
  store i8 %1, ptr %5, align 1
  store ptr %0, ptr %4, align 8
  %6 = load ptr, ptr %4, align 8
  ret ptr %6
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define internal noundef ptr @"??$?0V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@5@@Z@@?$_Tuple_val@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@QEAA@$$QEAV<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@1@@Z@QEAAXU?$coroutine_handle@X@1@@Z@@Z"(ptr noundef nonnull returned align 8 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(8) %1) unnamed_addr #0 align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr inbounds nuw %"struct.std::_Tuple_val", ptr %5, i32 0, i32 0
  %7 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %6, ptr align 8 %7, i64 8, i1 false)
  ret ptr %5
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define internal noundef ptr @"??$?0AEAPEAV?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@@?$_Compressed_pair@U?$default_delete@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@@std@@PEAV?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@2@$00@std@@QEAA@U_Zero_then_variadic_args_t@1@AEAPEAV?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@1@@Z"(ptr noundef nonnull returned align 8 dereferenceable(8) %0, i8 %1, ptr noundef nonnull align 8 dereferenceable(8) %2) unnamed_addr #0 align 2 {
  %4 = alloca %"struct.std::_Zero_then_variadic_args_t", align 1
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = getelementptr inbounds nuw %"struct.std::_Zero_then_variadic_args_t", ptr %4, i32 0, i32 0
  store i8 %1, ptr %7, align 1
  store ptr %2, ptr %5, align 8
  store ptr %0, ptr %6, align 8
  %8 = load ptr, ptr %6, align 8
  %9 = getelementptr inbounds nuw %"class.std::_Compressed_pair", ptr %8, i32 0, i32 0
  %10 = load ptr, ptr %5, align 8, !nonnull !16, !align !17
  %11 = load ptr, ptr %10, align 8
  store ptr %11, ptr %9, align 8
  ret ptr %8
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define internal void @"??$invoke@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@YAX$$QEAV<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@0@@Z@QEAAXU?$coroutine_handle@X@0@@Z@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0) #10 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8, !nonnull !16, !align !17
  call void @"??R<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@5@@Z@QEBA?A?<auto>@@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %3)
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define internal noundef nonnull align 8 dereferenceable(8) ptr @"??$get@$0A@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@YAAEAV<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@0@@Z@QEAAXU?$coroutine_handle@X@0@@Z@AEAV?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@0@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0) #0 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8, !nonnull !16, !align !17
  %4 = getelementptr inbounds nuw %"class.std::tuple", ptr %3, i32 0, i32 0
  %5 = getelementptr inbounds nuw %"struct.std::_Tuple_val", ptr %4, i32 0, i32 0
  ret ptr %5
}

declare dso_local void @_Cnd_do_broadcast_at_thread_exit() #14

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define internal void @"??R<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@5@@Z@QEBA?A?<auto>@@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) #10 align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %class.anon, ptr %3, i32 0, i32 0
  call void @"?resume@?$coroutine_handle@X@std@@QEBAXXZ"(ptr noundef nonnull align 8 dereferenceable(8) %4)
  ret void
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local void @"?resume@?$coroutine_handle@X@std@@QEBAXXZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) #10 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"struct.std::coroutine_handle", ptr %3, i32 0, i32 0
  %5 = load ptr, ptr %4, align 8
  call void @llvm.coro.resume(ptr %5)
  ret void
}

declare void @llvm.coro.resume(ptr)

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define internal noundef ptr @"??$exchange@PEAV?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@$$T@std@@YAPEAV?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@0@AEAPEAV10@$$QEA$$T@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(8) %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %6 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %7 = load ptr, ptr %6, align 8
  store ptr %7, ptr %5, align 8
  %8 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %9 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  store ptr null, ptr %9, align 8
  %10 = load ptr, ptr %5, align 8
  ret ptr %10
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define internal noundef nonnull align 1 dereferenceable(1) ptr @"?_Get_first@?$_Compressed_pair@U?$default_delete@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@@std@@PEAV?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@2@$00@std@@QEAAAEAU?$default_delete@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@@2@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) #0 align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret ptr %3
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define internal void @"??R?$default_delete@V?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@std@@@std@@QEBAXPEAV?$tuple@V<lambda_1>@?0??await_suspend@awaitable@?1??switch_to_new_thread@@YA@AEAVjthread@std@@@Z@QEAAXU?$coroutine_handle@X@6@@Z@@1@@Z"(ptr noundef nonnull align 1 dereferenceable(1) %0, ptr noundef %1) #0 align 2 personality ptr @__CxxFrameHandler3 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = load ptr, ptr %3, align 8
  %7 = icmp eq ptr %6, null
  br i1 %7, label %11, label %8

8:                                                ; preds = %2
  invoke void @llvm.seh.scope.begin()
          to label %9 unwind label %12

9:                                                ; preds = %8
  invoke void @llvm.seh.scope.end()
          to label %10 unwind label %12

10:                                               ; preds = %9
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %6, i64 noundef 8) #23
  br label %11

11:                                               ; preds = %10, %2
  ret void

12:                                               ; preds = %9, %8
  %13 = cleanuppad within none []
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %6, i64 noundef 8) #23 [ "funclet"(token %13) ]
  cleanupret from %13 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef i32 @"?fetch_sub@?$_Atomic_integral_facade@I@std@@QEAAIIW4memory_order@2@@Z"(ptr noundef nonnull align 4 dereferenceable(4) %0, i32 noundef %1, i32 noundef %2) #0 comdat align 2 {
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca ptr, align 8
  store i32 %2, ptr %4, align 4
  store i32 %1, ptr %5, align 4
  store ptr %0, ptr %6, align 8
  %7 = load ptr, ptr %6, align 8
  %8 = load i32, ptr %4, align 4
  %9 = load i32, ptr %5, align 4
  %10 = call noundef i32 @"?_Negate@?$_Atomic_integral_facade@I@std@@SAII@Z"(i32 noundef %9) #3
  %11 = call noundef i32 @"?fetch_add@?$_Atomic_integral@I$03@std@@QEAAIIW4memory_order@2@@Z"(ptr noundef nonnull align 4 dereferenceable(4) %7, i32 noundef %10, i32 noundef %8) #3
  ret i32 %11
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef i32 @"?fetch_add@?$_Atomic_integral@I$03@std@@QEAAIIW4memory_order@2@@Z"(ptr noundef nonnull align 4 dereferenceable(4) %0, i32 noundef %1, i32 noundef %2) #0 comdat align 2 {
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca ptr, align 8
  %7 = alloca i32, align 4
  store i32 %2, ptr %4, align 4
  store i32 %1, ptr %5, align 4
  store ptr %0, ptr %6, align 8
  %8 = load ptr, ptr %6, align 8
  %9 = load i32, ptr %4, align 4
  call void @_Check_memory_order(i32 noundef %9) #3
  %10 = getelementptr inbounds nuw %"struct.std::_Atomic_storage", ptr %8, i32 0, i32 0
  %11 = call noundef ptr @"??$_Atomic_address_as@JU?$_Atomic_padded@I@std@@@std@@YAPECJAEAU?$_Atomic_padded@I@0@@Z"(ptr noundef nonnull align 4 dereferenceable(4) %10) #3
  %12 = load i32, ptr %5, align 4
  %13 = atomicrmw add ptr %11, i32 %12 seq_cst, align 4
  store i32 %13, ptr %7, align 4
  %14 = load i32, ptr %7, align 4
  ret i32 %14
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef i32 @"?_Negate@?$_Atomic_integral_facade@I@std@@SAII@Z"(i32 noundef %0) #0 comdat align 2 {
  %2 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
  %3 = load i32, ptr %2, align 4
  %4 = sub i32 0, %3
  ret i32 %4
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @_Check_memory_order(i32 noundef %0) #0 comdat {
  %2 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
  %3 = load i32, ptr %2, align 4
  %4 = icmp ugt i32 %3, 5
  br i1 %4, label %5, label %6

5:                                                ; preds = %1
  br label %6

6:                                                ; preds = %5, %1
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??$_Atomic_address_as@JU?$_Atomic_padded@I@std@@@std@@YAPECJAEAU?$_Atomic_padded@I@0@@Z"(ptr noundef nonnull align 4 dereferenceable(4) %0) #0 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8, !nonnull !16, !align !23
  ret ptr %3
}

; Function Attrs: noreturn nounwind
declare dso_local void @terminate() #18

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?_Try_cancel_and_join@jthread@std@@AEAAXXZ"(ptr noundef nonnull align 8 dereferenceable(24) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"class.std::jthread", ptr %3, i32 0, i32 0
  %5 = call noundef zeroext i1 @"?joinable@thread@std@@QEBA_NXZ"(ptr noundef nonnull align 8 dereferenceable(16) %4) #3
  br i1 %5, label %6, label %10

6:                                                ; preds = %1
  %7 = getelementptr inbounds nuw %"class.std::jthread", ptr %3, i32 0, i32 1
  %8 = call noundef zeroext i1 @"?request_stop@stop_source@std@@QEAA_NXZ"(ptr noundef nonnull align 8 dereferenceable(8) %7) #3
  %9 = getelementptr inbounds nuw %"class.std::jthread", ptr %3, i32 0, i32 0
  call void @"?join@thread@std@@QEAAXXZ"(ptr noundef nonnull align 8 dereferenceable(16) %9)
  br label %10

10:                                               ; preds = %6, %1
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(16) ptr @"??4thread@std@@QEAAAEAV01@$$QEAV01@@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1) #0 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca %struct._Thrd_t, align 8
  %6 = alloca %struct._Thrd_t, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %7 = load ptr, ptr %4, align 8
  %8 = call noundef zeroext i1 @"?joinable@thread@std@@QEBA_NXZ"(ptr noundef nonnull align 8 dereferenceable(16) %7) #3
  br i1 %8, label %9, label %10

9:                                                ; preds = %2
  call void @terminate() #25
  unreachable

10:                                               ; preds = %2
  %11 = getelementptr inbounds nuw %struct._Thrd_t, ptr %6, i32 0, i32 0
  store ptr null, ptr %11, align 8
  %12 = getelementptr inbounds nuw %struct._Thrd_t, ptr %6, i32 0, i32 1
  store i32 0, ptr %12, align 8
  %13 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %14 = getelementptr inbounds nuw %"class.std::thread", ptr %13, i32 0, i32 0
  call void @"??$exchange@U_Thrd_t@@U1@@std@@YA?AU_Thrd_t@@AEAU1@$$QEAU1@@Z"(ptr dead_on_unwind writable sret(%struct._Thrd_t) align 8 %5, ptr noundef nonnull align 8 dereferenceable(16) %14, ptr noundef nonnull align 8 dereferenceable(16) %6) #3
  %15 = getelementptr inbounds nuw %"class.std::thread", ptr %7, i32 0, i32 0
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %15, ptr align 8 %5, i64 16, i1 false)
  ret ptr %7
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(8) ptr @"??4stop_source@std@@QEAAAEAV01@$$QEAV01@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(8) %1) #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca %"class.std::stop_source", align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %8 = call noundef ptr @"??0stop_source@std@@QEAA@$$QEAV01@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %5, ptr noundef nonnull align 8 dereferenceable(8) %7) #3
  invoke void @llvm.seh.scope.begin()
          to label %9 unwind label %11

9:                                                ; preds = %2
  call void @"?swap@stop_source@std@@QEAAXAEAV12@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %5, ptr noundef nonnull align 8 dereferenceable(8) %6) #3
  invoke void @llvm.seh.scope.end()
          to label %10 unwind label %11

10:                                               ; preds = %9
  call void @"??1stop_source@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %5) #3
  ret ptr %6

11:                                               ; preds = %9, %2
  %12 = cleanuppad within none []
  call void @"??1stop_source@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %5) #3 [ "funclet"(token %12) ]
  cleanupret from %12 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef zeroext i1 @"?request_stop@stop_source@std@@QEAA_NXZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr inbounds nuw %"class.std::stop_source", ptr %4, i32 0, i32 0
  %6 = load ptr, ptr %5, align 8
  store ptr %6, ptr %3, align 8
  %7 = load ptr, ptr %3, align 8
  %8 = icmp ne ptr %7, null
  br i1 %8, label %9, label %12

9:                                                ; preds = %1
  %10 = load ptr, ptr %3, align 8
  %11 = call noundef zeroext i1 @"?_Request_stop@_Stop_state@std@@QEAA_NXZ"(ptr noundef nonnull align 8 dereferenceable(32) %10) #3
  br label %12

12:                                               ; preds = %9, %1
  %13 = phi i1 [ false, %1 ], [ %11, %9 ]
  ret i1 %13
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local void @"?join@thread@std@@QEAAXXZ"(ptr noundef nonnull align 8 dereferenceable(16) %0) #10 comdat align 2 {
  %2 = alloca ptr, align 8
  %3 = alloca %struct._Thrd_t, align 8
  %4 = alloca %struct._Thrd_t, align 8
  store ptr %0, ptr %2, align 8
  %5 = load ptr, ptr %2, align 8
  %6 = call noundef zeroext i1 @"?joinable@thread@std@@QEBA_NXZ"(ptr noundef nonnull align 8 dereferenceable(16) %5) #3
  br i1 %6, label %8, label %7

7:                                                ; preds = %1
  call void @"?_Throw_Cpp_error@std@@YAXH@Z"(i32 noundef 1) #22
  unreachable

8:                                                ; preds = %1
  %9 = getelementptr inbounds nuw %"class.std::thread", ptr %5, i32 0, i32 0
  %10 = getelementptr inbounds nuw %struct._Thrd_t, ptr %9, i32 0, i32 1
  %11 = load i32, ptr %10, align 8
  %12 = call i32 @_Thrd_id()
  %13 = icmp eq i32 %11, %12
  br i1 %13, label %14, label %15

14:                                               ; preds = %8
  call void @"?_Throw_Cpp_error@std@@YAXH@Z"(i32 noundef 5) #22
  unreachable

15:                                               ; preds = %8
  %16 = getelementptr inbounds nuw %"class.std::thread", ptr %5, i32 0, i32 0
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %3, ptr align 8 %16, i64 16, i1 false)
  %17 = call i32 @_Thrd_join(ptr noundef %3, ptr noundef null)
  %18 = icmp ne i32 %17, 0
  br i1 %18, label %19, label %20

19:                                               ; preds = %15
  call void @"?_Throw_Cpp_error@std@@YAXH@Z"(i32 noundef 2) #22
  unreachable

20:                                               ; preds = %15
  %21 = getelementptr inbounds nuw %struct._Thrd_t, ptr %4, i32 0, i32 0
  store ptr null, ptr %21, align 8
  %22 = getelementptr inbounds nuw %struct._Thrd_t, ptr %4, i32 0, i32 1
  store i32 0, ptr %22, align 8
  %23 = getelementptr inbounds nuw %"class.std::thread", ptr %5, i32 0, i32 0
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %23, ptr align 8 %4, i64 16, i1 false)
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef zeroext i1 @"?_Request_stop@_Stop_state@std@@QEAA_NXZ"(ptr noundef nonnull align 8 dereferenceable(32) %0) #0 comdat align 2 {
  %2 = alloca i1, align 1
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  %7 = load ptr, ptr %3, align 8
  %8 = getelementptr inbounds nuw %"struct.std::_Stop_state", ptr %7, i32 0, i32 1
  %9 = call noundef i32 @"?fetch_or@?$_Atomic_integral@I$03@std@@QEAAIIW4memory_order@2@@Z"(ptr noundef nonnull align 4 dereferenceable(4) %8, i32 noundef 1, i32 noundef 5) #3
  %10 = and i32 %9, 1
  %11 = icmp ne i32 %10, 0
  br i1 %11, label %12, label %13

12:                                               ; preds = %1
  store i1 false, ptr %2, align 1
  br label %42

13:                                               ; preds = %1
  %14 = call i32 @_Thrd_id()
  %15 = getelementptr inbounds nuw %"struct.std::_Stop_state", ptr %7, i32 0, i32 4
  store i32 %14, ptr %15, align 8
  br label %16

16:                                               ; preds = %35, %13
  %17 = getelementptr inbounds nuw %"struct.std::_Stop_state", ptr %7, i32 0, i32 2
  %18 = call noundef ptr @"?_Lock_and_load@?$_Locked_pointer@V_Stop_callback_base@std@@@std@@QEAAPEAV_Stop_callback_base@2@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %17) #3
  store ptr %18, ptr %4, align 8
  %19 = getelementptr inbounds nuw %"struct.std::_Stop_state", ptr %7, i32 0, i32 3
  %20 = load ptr, ptr %4, align 8
  call void @"?store@?$_Atomic_storage@PEBV_Stop_callback_base@std@@$07@std@@QEAAXQEBV_Stop_callback_base@2@W4memory_order@2@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %19, ptr noundef %20, i32 noundef 0) #3
  %21 = getelementptr inbounds nuw %"struct.std::_Stop_state", ptr %7, i32 0, i32 3
  call void @"?notify_all@?$_Atomic_storage@PEBV_Stop_callback_base@std@@$07@std@@QEAAXXZ"(ptr noundef nonnull align 8 dereferenceable(8) %21) #3
  %22 = load ptr, ptr %4, align 8
  %23 = icmp eq ptr %22, null
  br i1 %23, label %24, label %26

24:                                               ; preds = %16
  %25 = getelementptr inbounds nuw %"struct.std::_Stop_state", ptr %7, i32 0, i32 2
  call void @"?_Store_and_unlock@?$_Locked_pointer@V_Stop_callback_base@std@@@std@@QEAAXQEAV_Stop_callback_base@2@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %25, ptr noundef null) #3
  store i1 true, ptr %2, align 1
  br label %42

26:                                               ; preds = %16
  store ptr null, ptr %6, align 8
  %27 = load ptr, ptr %4, align 8
  %28 = getelementptr inbounds nuw %"class.std::_Stop_callback_base", ptr %27, i32 0, i32 1
  %29 = call noundef ptr @"??$exchange@PEAV_Stop_callback_base@std@@$$T@std@@YAPEAV_Stop_callback_base@0@AEAPEAV10@$$QEA$$T@Z"(ptr noundef nonnull align 8 dereferenceable(8) %28, ptr noundef nonnull align 8 dereferenceable(8) %6) #3
  store ptr %29, ptr %5, align 8
  %30 = load ptr, ptr %5, align 8
  %31 = icmp ne ptr %30, null
  br i1 %31, label %32, label %35

32:                                               ; preds = %26
  %33 = load ptr, ptr %5, align 8
  %34 = getelementptr inbounds nuw %"class.std::_Stop_callback_base", ptr %33, i32 0, i32 2
  store ptr null, ptr %34, align 8
  br label %35

35:                                               ; preds = %32, %26
  %36 = getelementptr inbounds nuw %"struct.std::_Stop_state", ptr %7, i32 0, i32 2
  %37 = load ptr, ptr %5, align 8
  call void @"?_Store_and_unlock@?$_Locked_pointer@V_Stop_callback_base@std@@@std@@QEAAXQEAV_Stop_callback_base@2@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %36, ptr noundef %37) #3
  %38 = load ptr, ptr %4, align 8
  %39 = getelementptr inbounds nuw %"class.std::_Stop_callback_base", ptr %38, i32 0, i32 3
  %40 = load ptr, ptr %39, align 8
  %41 = load ptr, ptr %4, align 8
  call void %40(ptr noundef %41) #3
  br label %16, !llvm.loop !24

42:                                               ; preds = %24, %12
  %43 = load i1, ptr %2, align 1
  ret i1 %43
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef i32 @"?fetch_or@?$_Atomic_integral@I$03@std@@QEAAIIW4memory_order@2@@Z"(ptr noundef nonnull align 4 dereferenceable(4) %0, i32 noundef %1, i32 noundef %2) #0 comdat align 2 {
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca ptr, align 8
  %7 = alloca i32, align 4
  store i32 %2, ptr %4, align 4
  store i32 %1, ptr %5, align 4
  store ptr %0, ptr %6, align 8
  %8 = load ptr, ptr %6, align 8
  %9 = load i32, ptr %4, align 4
  call void @_Check_memory_order(i32 noundef %9) #3
  %10 = getelementptr inbounds nuw %"struct.std::_Atomic_storage", ptr %8, i32 0, i32 0
  %11 = call noundef ptr @"??$_Atomic_address_as@JU?$_Atomic_padded@I@std@@@std@@YAPECJAEAU?$_Atomic_padded@I@0@@Z"(ptr noundef nonnull align 4 dereferenceable(4) %10) #3
  %12 = load i32, ptr %5, align 4
  %13 = atomicrmw or ptr %11, i32 %12 seq_cst, align 4
  store i32 %13, ptr %7, align 4
  %14 = load i32, ptr %7, align 4
  ret i32 %14
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"?_Lock_and_load@?$_Locked_pointer@V_Stop_callback_base@std@@@std@@QEAAPEAV_Stop_callback_base@2@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  %3 = alloca i64, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr inbounds nuw %"class.std::_Locked_pointer", ptr %4, i32 0, i32 0
  %6 = call noundef i64 @"?load@?$_Atomic_storage@_K$07@std@@QEBA_KW4memory_order@2@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %5, i32 noundef 0) #3
  store i64 %6, ptr %3, align 8
  br label %7

7:                                                ; preds = %36, %1
  %8 = load i64, ptr %3, align 8
  %9 = and i64 %8, 3
  switch i64 %9, label %35 [
    i64 0, label %10
    i64 1, label %19
    i64 2, label %30
  ]

10:                                               ; preds = %7
  %11 = getelementptr inbounds nuw %"class.std::_Locked_pointer", ptr %4, i32 0, i32 0
  %12 = load i64, ptr %3, align 8
  %13 = or i64 %12, 1
  %14 = call noundef zeroext i1 @"?compare_exchange_weak@?$atomic@_K@std@@QEAA_NAEA_K_K@Z"(ptr noundef nonnull align 8 dereferenceable(8) %11, ptr noundef nonnull align 8 dereferenceable(8) %3, i64 noundef %13) #3
  br i1 %14, label %15, label %18

15:                                               ; preds = %10
  %16 = load i64, ptr %3, align 8
  %17 = inttoptr i64 %16 to ptr
  ret ptr %17

18:                                               ; preds = %10
  call void @llvm.x86.sse2.pause()
  br label %36

19:                                               ; preds = %7
  %20 = getelementptr inbounds nuw %"class.std::_Locked_pointer", ptr %4, i32 0, i32 0
  %21 = load i64, ptr %3, align 8
  %22 = and i64 %21, -4
  %23 = or i64 %22, 2
  %24 = call noundef zeroext i1 @"?compare_exchange_weak@?$atomic@_K@std@@QEAA_NAEA_K_K@Z"(ptr noundef nonnull align 8 dereferenceable(8) %20, ptr noundef nonnull align 8 dereferenceable(8) %3, i64 noundef %23) #3
  br i1 %24, label %26, label %25

25:                                               ; preds = %19
  call void @llvm.x86.sse2.pause()
  br label %36

26:                                               ; preds = %19
  %27 = load i64, ptr %3, align 8
  %28 = and i64 %27, -4
  %29 = or i64 %28, 2
  store i64 %29, ptr %3, align 8
  br label %30

30:                                               ; preds = %7, %26
  %31 = getelementptr inbounds nuw %"class.std::_Locked_pointer", ptr %4, i32 0, i32 0
  %32 = load i64, ptr %3, align 8
  call void @"?wait@?$_Atomic_storage@_K$07@std@@QEBAX_KW4memory_order@2@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %31, i64 noundef %32, i32 noundef 0) #3
  %33 = getelementptr inbounds nuw %"class.std::_Locked_pointer", ptr %4, i32 0, i32 0
  %34 = call noundef i64 @"?load@?$_Atomic_storage@_K$07@std@@QEBA_KW4memory_order@2@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %33, i32 noundef 0) #3
  store i64 %34, ptr %3, align 8
  br label %36

35:                                               ; preds = %7
  call void @abort() #25
  unreachable

36:                                               ; preds = %30, %25, %18
  br label %7, !llvm.loop !25
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?store@?$_Atomic_storage@PEBV_Stop_callback_base@std@@$07@std@@QEAAXQEBV_Stop_callback_base@2@W4memory_order@2@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0, ptr noundef %1, i32 noundef %2) #0 comdat align 2 {
  %4 = alloca i32, align 4
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca i64, align 8
  store i32 %2, ptr %4, align 4
  store ptr %1, ptr %5, align 8
  store ptr %0, ptr %6, align 8
  %9 = load ptr, ptr %6, align 8
  %10 = getelementptr inbounds nuw %"struct.std::_Atomic_storage.7", ptr %9, i32 0, i32 0
  %11 = call noundef ptr @"??$_Atomic_address_as@_JU?$_Atomic_padded@PEBV_Stop_callback_base@std@@@std@@@std@@YAPEC_JAEAU?$_Atomic_padded@PEBV_Stop_callback_base@std@@@0@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %10) #3
  store ptr %11, ptr %7, align 8
  %12 = call noundef i64 @"??$_Atomic_reinterpret_as@_JPEBV_Stop_callback_base@std@@@std@@YA_JAEBQEBV_Stop_callback_base@0@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %5) #3
  store i64 %12, ptr %8, align 8
  %13 = load i32, ptr %4, align 4
  switch i32 %13, label %20 [
    i32 0, label %14
    i32 3, label %17
    i32 1, label %21
    i32 2, label %21
    i32 4, label %21
    i32 5, label %22
  ]

14:                                               ; preds = %3
  %15 = load ptr, ptr %7, align 8
  %16 = load i64, ptr %8, align 8
  store volatile i64 %16, ptr %15, align 8
  br label %24

17:                                               ; preds = %3
  fence syncscope("singlethread") seq_cst
  %18 = load ptr, ptr %7, align 8
  %19 = load i64, ptr %8, align 8
  store volatile i64 %19, ptr %18, align 8
  br label %24

20:                                               ; preds = %3
  br label %21

21:                                               ; preds = %3, %3, %3, %20
  br label %22

22:                                               ; preds = %3, %21
  %23 = load ptr, ptr %5, align 8
  call void @"?store@?$_Atomic_storage@PEBV_Stop_callback_base@std@@$07@std@@QEAAXQEBV_Stop_callback_base@2@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %9, ptr noundef %23) #3
  br label %24

24:                                               ; preds = %22, %17, %14
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?notify_all@?$_Atomic_storage@PEBV_Stop_callback_base@std@@$07@std@@QEAAXXZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"struct.std::_Atomic_storage.7", ptr %3, i32 0, i32 0
  call void @__std_atomic_notify_all_direct(ptr noundef %4) #3
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?_Store_and_unlock@?$_Locked_pointer@V_Stop_callback_base@std@@@std@@QEAAXQEAV_Stop_callback_base@2@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0, ptr noundef %1) #0 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i64, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = getelementptr inbounds nuw %"class.std::_Locked_pointer", ptr %6, i32 0, i32 0
  %8 = load ptr, ptr %3, align 8
  %9 = ptrtoint ptr %8 to i64
  %10 = call noundef i64 @"?exchange@?$_Atomic_storage@_K$07@std@@QEAA_K_KW4memory_order@2@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %7, i64 noundef %9, i32 noundef 5) #3
  store i64 %10, ptr %5, align 8
  %11 = load i64, ptr %5, align 8
  %12 = and i64 %11, 3
  %13 = icmp eq i64 %12, 2
  br i1 %13, label %14, label %16

14:                                               ; preds = %2
  %15 = getelementptr inbounds nuw %"class.std::_Locked_pointer", ptr %6, i32 0, i32 0
  call void @"?notify_all@?$_Atomic_storage@_K$07@std@@QEAAXXZ"(ptr noundef nonnull align 8 dereferenceable(8) %15) #3
  br label %16

16:                                               ; preds = %14, %2
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??$exchange@PEAV_Stop_callback_base@std@@$$T@std@@YAPEAV_Stop_callback_base@0@AEAPEAV10@$$QEA$$T@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(8) %1) #0 comdat {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %6 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %7 = load ptr, ptr %6, align 8
  store ptr %7, ptr %5, align 8
  %8 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %9 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  store ptr null, ptr %9, align 8
  %10 = load ptr, ptr %5, align 8
  ret ptr %10
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef i64 @"?load@?$_Atomic_storage@_K$07@std@@QEBA_KW4memory_order@2@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0, i32 noundef %1) #0 comdat align 2 {
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i64, align 8
  store i32 %1, ptr %3, align 4
  store ptr %0, ptr %4, align 8
  %7 = load ptr, ptr %4, align 8
  %8 = getelementptr inbounds nuw %"struct.std::_Atomic_storage.4", ptr %7, i32 0, i32 0
  %9 = call noundef ptr @"??$_Atomic_address_as@_JU?$_Atomic_padded@_K@std@@@std@@YAPED_JAEBU?$_Atomic_padded@_K@0@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %8) #3
  store ptr %9, ptr %5, align 8
  %10 = load ptr, ptr %5, align 8
  %11 = load volatile i64, ptr %10, align 8
  store i64 %11, ptr %6, align 8
  %12 = load i32, ptr %3, align 4
  switch i32 %12, label %16 [
    i32 0, label %13
    i32 1, label %14
    i32 2, label %14
    i32 5, label %14
    i32 3, label %15
    i32 4, label %15
  ]

13:                                               ; preds = %2
  br label %17

14:                                               ; preds = %2, %2, %2
  fence syncscope("singlethread") seq_cst
  br label %17

15:                                               ; preds = %2, %2
  br label %16

16:                                               ; preds = %2, %15
  br label %17

17:                                               ; preds = %16, %14, %13
  %18 = load i64, ptr %6, align 8
  ret i64 %18
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef zeroext i1 @"?compare_exchange_weak@?$atomic@_K@std@@QEAA_NAEA_K_K@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(8) %1, i64 noundef %2) #0 comdat align 2 {
  %4 = alloca i64, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  store i64 %2, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store ptr %0, ptr %6, align 8
  %7 = load ptr, ptr %6, align 8
  %8 = load i64, ptr %4, align 8
  %9 = load ptr, ptr %5, align 8, !nonnull !16, !align !17
  %10 = call noundef zeroext i1 @"?compare_exchange_strong@?$_Atomic_storage@_K$07@std@@QEAA_NAEA_K_KW4memory_order@2@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %7, ptr noundef nonnull align 8 dereferenceable(8) %9, i64 noundef %8, i32 noundef 5) #3
  ret i1 %10
}

; Function Attrs: nounwind
declare void @llvm.x86.sse2.pause() #3

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?wait@?$_Atomic_storage@_K$07@std@@QEBAX_KW4memory_order@2@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0, i64 noundef %1, i32 noundef %2) #0 comdat align 2 {
  %4 = alloca i32, align 4
  %5 = alloca i64, align 8
  %6 = alloca ptr, align 8
  store i32 %2, ptr %4, align 4
  store i64 %1, ptr %5, align 8
  store ptr %0, ptr %6, align 8
  %7 = load ptr, ptr %6, align 8
  %8 = load i32, ptr %4, align 4
  %9 = call noundef i64 @"??$_Atomic_reinterpret_as@_J_K@std@@YA_JAEB_K@Z"(ptr noundef nonnull align 8 dereferenceable(8) %5) #3
  call void @"??$_Atomic_wait_direct@_K_J@std@@YAXQEBU?$_Atomic_storage@_K$07@0@_JW4memory_order@0@@Z"(ptr noundef %7, i64 noundef %9, i32 noundef %8) #3
  ret void
}

; Function Attrs: noreturn nounwind
declare dso_local void @abort() #18

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??$_Atomic_address_as@_JU?$_Atomic_padded@_K@std@@@std@@YAPED_JAEBU?$_Atomic_padded@_K@0@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0) #0 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8, !nonnull !16, !align !17
  ret ptr %3
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef zeroext i1 @"?compare_exchange_strong@?$_Atomic_storage@_K$07@std@@QEAA_NAEA_K_KW4memory_order@2@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(8) %1, i64 noundef %2, i32 noundef %3) #0 comdat align 2 {
  %5 = alloca i1, align 1
  %6 = alloca i32, align 4
  %7 = alloca i64, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca i64, align 8
  %11 = alloca i64, align 8
  store i32 %3, ptr %6, align 4
  store i64 %2, ptr %7, align 8
  store ptr %1, ptr %8, align 8
  store ptr %0, ptr %9, align 8
  %12 = load ptr, ptr %9, align 8
  %13 = load ptr, ptr %8, align 8, !nonnull !16, !align !17
  %14 = call noundef i64 @"??$_Atomic_reinterpret_as@_J_K@std@@YA_JAEB_K@Z"(ptr noundef nonnull align 8 dereferenceable(8) %13) #3
  store i64 %14, ptr %10, align 8
  %15 = load i32, ptr %6, align 4
  call void @_Check_memory_order(i32 noundef %15) #3
  %16 = getelementptr inbounds nuw %"struct.std::_Atomic_storage.4", ptr %12, i32 0, i32 0
  %17 = call noundef ptr @"??$_Atomic_address_as@_JU?$_Atomic_padded@_K@std@@@std@@YAPEC_JAEAU?$_Atomic_padded@_K@0@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %16) #3
  %18 = call noundef i64 @"??$_Atomic_reinterpret_as@_J_K@std@@YA_JAEB_K@Z"(ptr noundef nonnull align 8 dereferenceable(8) %7) #3
  %19 = load i64, ptr %10, align 8
  %20 = cmpxchg volatile ptr %17, i64 %19, i64 %18 seq_cst seq_cst, align 8
  %21 = extractvalue { i64, i1 } %20, 0
  store i64 %21, ptr %11, align 8
  %22 = load i64, ptr %11, align 8
  %23 = load i64, ptr %10, align 8
  %24 = icmp eq i64 %22, %23
  br i1 %24, label %25, label %26

25:                                               ; preds = %4
  store i1 true, ptr %5, align 1
  br label %28

26:                                               ; preds = %4
  %27 = load ptr, ptr %8, align 8, !nonnull !16, !align !17
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %27, ptr align 8 %11, i64 8, i1 false)
  store i1 false, ptr %5, align 1
  br label %28

28:                                               ; preds = %26, %25
  %29 = load i1, ptr %5, align 1
  ret i1 %29
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef i64 @"??$_Atomic_reinterpret_as@_J_K@std@@YA_JAEB_K@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0) #0 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8, !nonnull !16, !align !17
  %4 = load i64, ptr %3, align 8
  ret i64 %4
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??$_Atomic_address_as@_JU?$_Atomic_padded@_K@std@@@std@@YAPEC_JAEAU?$_Atomic_padded@_K@0@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0) #0 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8, !nonnull !16, !align !17
  ret ptr %3
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"??$_Atomic_wait_direct@_K_J@std@@YAXQEBU?$_Atomic_storage@_K$07@0@_JW4memory_order@0@@Z"(ptr noundef %0, i64 noundef %1, i32 noundef %2) #0 comdat {
  %4 = alloca i32, align 4
  %5 = alloca i64, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca i64, align 8
  %9 = alloca i64, align 8
  store i32 %2, ptr %4, align 4
  store i64 %1, ptr %5, align 8
  store ptr %0, ptr %6, align 8
  %10 = load ptr, ptr %6, align 8
  %11 = getelementptr inbounds nuw %"struct.std::_Atomic_storage.4", ptr %10, i32 0, i32 0
  store ptr %11, ptr %7, align 8
  br label %12

12:                                               ; preds = %21, %3
  %13 = load ptr, ptr %6, align 8
  %14 = load i32, ptr %4, align 4
  %15 = call noundef i64 @"?load@?$_Atomic_storage@_K$07@std@@QEBA_KW4memory_order@2@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %13, i32 noundef %14) #3
  store i64 %15, ptr %9, align 8
  %16 = call noundef i64 @"??$_Atomic_reinterpret_as@_J_K@std@@YA_JAEB_K@Z"(ptr noundef nonnull align 8 dereferenceable(8) %9) #3
  store i64 %16, ptr %8, align 8
  %17 = load i64, ptr %5, align 8
  %18 = load i64, ptr %8, align 8
  %19 = icmp ne i64 %17, %18
  br i1 %19, label %20, label %21

20:                                               ; preds = %12
  ret void

21:                                               ; preds = %12
  %22 = load ptr, ptr %7, align 8
  %23 = call i32 @__std_atomic_wait_direct(ptr noundef %22, ptr noundef %5, i64 noundef 8, i32 noundef -1) #3
  br label %12, !llvm.loop !26
}

; Function Attrs: nounwind
declare dso_local i32 @__std_atomic_wait_direct(ptr noundef, ptr noundef, i64 noundef, i32 noundef) #19

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??$_Atomic_address_as@_JU?$_Atomic_padded@PEBV_Stop_callback_base@std@@@std@@@std@@YAPEC_JAEAU?$_Atomic_padded@PEBV_Stop_callback_base@std@@@0@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0) #0 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8, !nonnull !16, !align !17
  ret ptr %3
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef i64 @"??$_Atomic_reinterpret_as@_JPEBV_Stop_callback_base@std@@@std@@YA_JAEBQEBV_Stop_callback_base@0@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0) #0 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8, !nonnull !16, !align !17
  %4 = load ptr, ptr %3, align 8
  %5 = ptrtoint ptr %4 to i64
  ret i64 %5
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?store@?$_Atomic_storage@PEBV_Stop_callback_base@std@@$07@std@@QEAAXQEBV_Stop_callback_base@2@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0, ptr noundef %1) #0 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i64, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %7 = load ptr, ptr %4, align 8
  %8 = getelementptr inbounds nuw %"struct.std::_Atomic_storage.7", ptr %7, i32 0, i32 0
  %9 = call noundef ptr @"??$_Atomic_address_as@_JU?$_Atomic_padded@PEBV_Stop_callback_base@std@@@std@@@std@@YAPEC_JAEAU?$_Atomic_padded@PEBV_Stop_callback_base@std@@@0@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %8) #3
  store ptr %9, ptr %5, align 8
  %10 = call noundef i64 @"??$_Atomic_reinterpret_as@_JPEBV_Stop_callback_base@std@@@std@@YA_JAEBQEBV_Stop_callback_base@0@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %3) #3
  store i64 %10, ptr %6, align 8
  %11 = load ptr, ptr %5, align 8
  %12 = load i64, ptr %6, align 8
  %13 = atomicrmw xchg ptr %11, i64 %12 seq_cst, align 8
  ret void
}

; Function Attrs: nounwind
declare dso_local void @__std_atomic_notify_all_direct(ptr noundef) #19

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef i64 @"?exchange@?$_Atomic_storage@_K$07@std@@QEAA_K_KW4memory_order@2@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0, i64 noundef %1, i32 noundef %2) #0 comdat align 2 {
  %4 = alloca i32, align 4
  %5 = alloca i64, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i64, align 8
  store i32 %2, ptr %4, align 4
  store i64 %1, ptr %5, align 8
  store ptr %0, ptr %6, align 8
  %8 = load ptr, ptr %6, align 8
  %9 = load i32, ptr %4, align 4
  call void @_Check_memory_order(i32 noundef %9) #3
  %10 = getelementptr inbounds nuw %"struct.std::_Atomic_storage.4", ptr %8, i32 0, i32 0
  %11 = call noundef ptr @"??$_Atomic_address_as@_JU?$_Atomic_padded@_K@std@@@std@@YAPEC_JAEAU?$_Atomic_padded@_K@0@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %10) #3
  %12 = call noundef i64 @"??$_Atomic_reinterpret_as@_J_K@std@@YA_JAEB_K@Z"(ptr noundef nonnull align 8 dereferenceable(8) %5) #3
  %13 = atomicrmw xchg ptr %11, i64 %12 seq_cst, align 8
  store i64 %13, ptr %7, align 8
  %14 = load i64, ptr %7, align 8
  ret i64 %14
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?notify_all@?$_Atomic_storage@_K$07@std@@QEAAXXZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"struct.std::_Atomic_storage.4", ptr %3, i32 0, i32 0
  call void @__std_atomic_notify_all_direct(ptr noundef %4) #3
  ret void
}

declare dso_local i32 @_Thrd_join(ptr noundef, ptr noundef) #14

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"??$exchange@U_Thrd_t@@U1@@std@@YA?AU_Thrd_t@@AEAU1@$$QEAU1@@Z"(ptr dead_on_unwind noalias writable sret(%struct._Thrd_t) align 8 %0, ptr noundef nonnull align 8 dereferenceable(16) %1, ptr noundef nonnull align 8 dereferenceable(16) %2) #0 comdat {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %2, ptr %5, align 8
  store ptr %1, ptr %6, align 8
  %7 = load ptr, ptr %6, align 8, !nonnull !16, !align !17
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 %7, i64 16, i1 false)
  %8 = load ptr, ptr %5, align 8, !nonnull !16, !align !17
  %9 = load ptr, ptr %6, align 8, !nonnull !16, !align !17
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %9, ptr align 8 %8, i64 16, i1 false)
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0stop_source@std@@QEAA@$$QEAV01@@Z"(ptr noundef nonnull returned align 8 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(8) %1) unnamed_addr #0 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = getelementptr inbounds nuw %"class.std::stop_source", ptr %6, i32 0, i32 0
  store ptr null, ptr %5, align 8
  %8 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %9 = getelementptr inbounds nuw %"class.std::stop_source", ptr %8, i32 0, i32 0
  %10 = call noundef ptr @"??$exchange@PEAU_Stop_state@std@@$$T@std@@YAPEAU_Stop_state@0@AEAPEAU10@$$QEA$$T@Z"(ptr noundef nonnull align 8 dereferenceable(8) %9, ptr noundef nonnull align 8 dereferenceable(8) %5) #3
  store ptr %10, ptr %7, align 8
  ret ptr %6
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?swap@stop_source@std@@QEAAXAEAV12@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(8) %1) #0 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %7 = getelementptr inbounds nuw %"class.std::stop_source", ptr %6, i32 0, i32 0
  %8 = getelementptr inbounds nuw %"class.std::stop_source", ptr %5, i32 0, i32 0
  call void @"??$swap@PEAU_Stop_state@std@@$0A@@std@@YAXAEAPEAU_Stop_state@0@0@Z"(ptr noundef nonnull align 8 dereferenceable(8) %8, ptr noundef nonnull align 8 dereferenceable(8) %7) #3
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??$exchange@PEAU_Stop_state@std@@$$T@std@@YAPEAU_Stop_state@0@AEAPEAU10@$$QEA$$T@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(8) %1) #0 comdat {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %6 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %7 = load ptr, ptr %6, align 8
  store ptr %7, ptr %5, align 8
  %8 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %9 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  store ptr null, ptr %9, align 8
  %10 = load ptr, ptr %5, align 8
  ret ptr %10
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"??$swap@PEAU_Stop_state@std@@$0A@@std@@YAXAEAPEAU_Stop_state@0@0@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(8) %1) #0 comdat {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %6 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %7 = load ptr, ptr %6, align 8
  store ptr %7, ptr %5, align 8
  %8 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %9 = load ptr, ptr %8, align 8
  %10 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  store ptr %9, ptr %10, align 8
  %11 = load ptr, ptr %5, align 8
  %12 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  store ptr %11, ptr %12, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?get_id@thread@std@@QEBA?AVid@12@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr dead_on_unwind noalias writable sret(%"class.std::thread::id") align 4 %1) #0 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr inbounds nuw %"class.std::thread", ptr %5, i32 0, i32 0
  %7 = getelementptr inbounds nuw %struct._Thrd_t, ptr %6, i32 0, i32 1
  %8 = load i32, ptr %7, align 8
  %9 = call noundef ptr @"??0id@thread@std@@AEAA@I@Z"(ptr noundef nonnull align 4 dereferenceable(4) %1, i32 noundef %8) #3
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0stop_source@std@@QEAA@Unostopstate_t@1@@Z"(ptr noundef nonnull returned align 8 dereferenceable(8) %0, i8 %1) unnamed_addr #0 comdat align 2 {
  %3 = alloca %"struct.std::nostopstate_t", align 1
  %4 = alloca ptr, align 8
  %5 = getelementptr inbounds nuw %"struct.std::nostopstate_t", ptr %3, i32 0, i32 0
  store i8 %1, ptr %5, align 1
  store ptr %0, ptr %4, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = getelementptr inbounds nuw %"class.std::stop_source", ptr %6, i32 0, i32 0
  store ptr null, ptr %7, align 8
  ret ptr %6
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef i64 @"?length@?$_Narrow_char_traits@DH@std@@SA_KQEBD@Z"(ptr noundef %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call i64 @strlen(ptr noundef %3) #3
  ret i64 %4
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef i64 @"?width@ios_base@std@@QEBA_JXZ"(ptr noundef nonnull align 8 dereferenceable(72) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"class.std::ios_base", ptr %3, i32 0, i32 6
  %5 = load i64, ptr %4, align 8
  ret i64 %5
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z"(ptr noundef nonnull returned align 8 dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(8) %1) unnamed_addr #10 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  store ptr %1, ptr %4, align 8
  store ptr %0, ptr %5, align 8
  %7 = load ptr, ptr %5, align 8
  store ptr %7, ptr %3, align 8
  %8 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %9 = call noundef ptr @"??0_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %7, ptr noundef nonnull align 8 dereferenceable(8) %8)
  invoke void @llvm.seh.scope.begin()
          to label %10 unwind label %57

10:                                               ; preds = %2
  %11 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %12 = getelementptr inbounds i8, ptr %11, i64 0
  %13 = load ptr, ptr %12, align 8
  %14 = getelementptr inbounds i32, ptr %13, i32 1
  %15 = load i32, ptr %14, align 4
  %16 = sext i32 %15 to i64
  %17 = add nsw i64 0, %16
  %18 = getelementptr inbounds i8, ptr %11, i64 %17
  %19 = call noundef zeroext i1 @"?good@ios_base@std@@QEBA_NXZ"(ptr noundef nonnull align 8 dereferenceable(72) %18) #3
  br i1 %19, label %22, label %20

20:                                               ; preds = %10
  %21 = getelementptr inbounds nuw %"class.std::basic_ostream<char>::sentry", ptr %7, i32 0, i32 1
  store i8 0, ptr %21, align 8
  br label %55

22:                                               ; preds = %10
  %23 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %24 = getelementptr inbounds i8, ptr %23, i64 0
  %25 = load ptr, ptr %24, align 8
  %26 = getelementptr inbounds i32, ptr %25, i32 1
  %27 = load i32, ptr %26, align 4
  %28 = sext i32 %27 to i64
  %29 = add nsw i64 0, %28
  %30 = getelementptr inbounds i8, ptr %23, i64 %29
  %31 = call noundef ptr @"?tie@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_ostream@DU?$char_traits@D@std@@@2@XZ"(ptr noundef nonnull align 8 dereferenceable(96) %30) #3
  store ptr %31, ptr %6, align 8
  %32 = load ptr, ptr %6, align 8
  %33 = icmp ne ptr %32, null
  br i1 %33, label %34, label %38

34:                                               ; preds = %22
  %35 = load ptr, ptr %6, align 8
  %36 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %37 = icmp eq ptr %35, %36
  br i1 %37, label %38, label %40

38:                                               ; preds = %34, %22
  %39 = getelementptr inbounds nuw %"class.std::basic_ostream<char>::sentry", ptr %7, i32 0, i32 1
  store i8 1, ptr %39, align 8
  br label %55

40:                                               ; preds = %34
  %41 = load ptr, ptr %6, align 8
  %42 = invoke noundef nonnull align 8 dereferenceable(8) ptr @"?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %41)
          to label %43 unwind label %57

43:                                               ; preds = %40
  %44 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %45 = getelementptr inbounds i8, ptr %44, i64 0
  %46 = load ptr, ptr %45, align 8
  %47 = getelementptr inbounds i32, ptr %46, i32 1
  %48 = load i32, ptr %47, align 4
  %49 = sext i32 %48 to i64
  %50 = add nsw i64 0, %49
  %51 = getelementptr inbounds i8, ptr %44, i64 %50
  %52 = call noundef zeroext i1 @"?good@ios_base@std@@QEBA_NXZ"(ptr noundef nonnull align 8 dereferenceable(72) %51) #3
  %53 = getelementptr inbounds nuw %"class.std::basic_ostream<char>::sentry", ptr %7, i32 0, i32 1
  %54 = zext i1 %52 to i8
  store i8 %54, ptr %53, align 8
  invoke void @llvm.seh.scope.end()
          to label %55 unwind label %57

55:                                               ; preds = %20, %38, %43
  %56 = load ptr, ptr %3, align 8
  ret ptr %56

57:                                               ; preds = %43, %40, %2
  %58 = cleanuppad within none []
  call void @"??1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %7) #3 [ "funclet"(token %58) ]
  cleanupret from %58 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef zeroext i1 @"??Bsentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEBA_NXZ"(ptr noundef nonnull align 8 dereferenceable(16) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"class.std::basic_ostream<char>::sentry", ptr %3, i32 0, i32 1
  %5 = load i8, ptr %4, align 8
  %6 = trunc i8 %5 to i1
  ret i1 %6
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef i32 @"?flags@ios_base@std@@QEBAHXZ"(ptr noundef nonnull align 8 dereferenceable(72) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"class.std::ios_base", ptr %3, i32 0, i32 4
  %5 = load i32, ptr %4, align 8
  ret i32 %5
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef zeroext i1 @"?eq_int_type@?$_Narrow_char_traits@DH@std@@SA_NHH@Z"(i32 noundef %0, i32 noundef %1) #0 comdat align 2 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 %1, ptr %3, align 4
  store i32 %0, ptr %4, align 4
  %5 = load i32, ptr %4, align 4
  %6 = load i32, ptr %3, align 4
  %7 = icmp eq i32 %5, %6
  ret i1 %7
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"?rdbuf@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_streambuf@DU?$char_traits@D@std@@@2@XZ"(ptr noundef nonnull align 8 dereferenceable(96) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"class.std::basic_ios", ptr %3, i32 0, i32 1
  %5 = load ptr, ptr %4, align 8
  ret ptr %5
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local noundef i32 @"?sputc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHD@Z"(ptr noundef nonnull align 8 dereferenceable(104) %0, i8 noundef %1) #10 comdat align 2 {
  %3 = alloca i8, align 1
  %4 = alloca ptr, align 8
  store i8 %1, ptr %3, align 1
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = call noundef i64 @"?_Pnavail@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEBA_JXZ"(ptr noundef nonnull align 8 dereferenceable(104) %5) #3
  %7 = icmp slt i64 0, %6
  br i1 %7, label %8, label %12

8:                                                ; preds = %2
  %9 = load i8, ptr %3, align 1
  %10 = call noundef ptr @"?_Pninc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAPEADXZ"(ptr noundef nonnull align 8 dereferenceable(104) %5) #3
  store i8 %9, ptr %10, align 1
  %11 = call noundef i32 @"?to_int_type@?$_Narrow_char_traits@DH@std@@SAHD@Z"(i8 noundef %9) #3
  br label %19

12:                                               ; preds = %2
  %13 = load i8, ptr %3, align 1
  %14 = call noundef i32 @"?to_int_type@?$_Narrow_char_traits@DH@std@@SAHD@Z"(i8 noundef %13) #3
  %15 = load ptr, ptr %5, align 8
  %16 = getelementptr inbounds ptr, ptr %15, i64 3
  %17 = load ptr, ptr %16, align 8
  %18 = call noundef i32 %17(ptr noundef nonnull align 8 dereferenceable(104) %5, i32 noundef %14)
  br label %19

19:                                               ; preds = %12, %8
  %20 = phi i32 [ %11, %8 ], [ %18, %12 ]
  ret i32 %20
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef i8 @"?fill@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADXZ"(ptr noundef nonnull align 8 dereferenceable(96) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"class.std::basic_ios", ptr %3, i32 0, i32 3
  %5 = load i8, ptr %4, align 8
  ret i8 %5
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef i32 @"?eof@?$_Narrow_char_traits@DH@std@@SAHXZ"() #0 comdat align 2 {
  ret i32 -1
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local noundef i64 @"?sputn@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAA_JPEBD_J@Z"(ptr noundef nonnull align 8 dereferenceable(104) %0, ptr noundef %1, i64 noundef %2) #10 comdat align 2 {
  %4 = alloca i64, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  store i64 %2, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store ptr %0, ptr %6, align 8
  %7 = load ptr, ptr %6, align 8
  %8 = load i64, ptr %4, align 8
  %9 = load ptr, ptr %5, align 8
  %10 = load ptr, ptr %7, align 8
  %11 = getelementptr inbounds ptr, ptr %10, i64 9
  %12 = load ptr, ptr %11, align 8
  %13 = call noundef i64 %12(ptr noundef nonnull align 8 dereferenceable(104) %7, ptr noundef %9, i64 noundef %8)
  ret i64 %13
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef i64 @"?width@ios_base@std@@QEAA_J_J@Z"(ptr noundef nonnull align 8 dereferenceable(72) %0, i64 noundef %1) #0 comdat align 2 {
  %3 = alloca i64, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i64, align 8
  store i64 %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = getelementptr inbounds nuw %"class.std::ios_base", ptr %6, i32 0, i32 6
  %8 = load i64, ptr %7, align 8
  store i64 %8, ptr %5, align 8
  %9 = load i64, ptr %3, align 8
  %10 = getelementptr inbounds nuw %"class.std::ios_base", ptr %6, i32 0, i32 6
  store i64 %9, ptr %10, align 8
  %11 = load i64, ptr %5, align 8
  ret i64 %11
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local void @"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"(ptr noundef nonnull align 8 dereferenceable(96) %0, i32 noundef %1, i1 noundef zeroext %2) #10 comdat align 2 {
  %4 = alloca i8, align 1
  %5 = alloca i32, align 4
  %6 = alloca ptr, align 8
  %7 = zext i1 %2 to i8
  store i8 %7, ptr %4, align 1
  store i32 %1, ptr %5, align 4
  store ptr %0, ptr %6, align 8
  %8 = load ptr, ptr %6, align 8
  %9 = load i8, ptr %4, align 1
  %10 = trunc i8 %9 to i1
  %11 = call noundef i32 @"?rdstate@ios_base@std@@QEBAHXZ"(ptr noundef nonnull align 8 dereferenceable(72) %8) #3
  %12 = load i32, ptr %5, align 4
  %13 = or i32 %11, %12
  call void @"?clear@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"(ptr noundef nonnull align 8 dereferenceable(96) %8, i32 noundef %13, i1 noundef zeroext %10)
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %0) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %2 = alloca ptr, align 8
  %3 = alloca i8, align 1
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  invoke void @llvm.seh.scope.begin()
          to label %5 unwind label %16

5:                                                ; preds = %1
  %6 = call noundef i32 @"?uncaught_exceptions@std@@YAHXZ"() #3
  %7 = icmp eq i32 %6, 0
  %8 = zext i1 %7 to i8
  store i8 %8, ptr %3, align 1
  %9 = load i8, ptr %3, align 1
  %10 = trunc i8 %9 to i1
  br i1 %10, label %11, label %14

11:                                               ; preds = %5
  %12 = getelementptr inbounds nuw %"class.std::basic_ostream<char>::_Sentry_base", ptr %4, i32 0, i32 0
  %13 = load ptr, ptr %12, align 8, !nonnull !16, !align !17
  call void @"?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ"(ptr noundef nonnull align 8 dereferenceable(8) %13) #3
  br label %14

14:                                               ; preds = %11, %5
  invoke void @llvm.seh.scope.end()
          to label %15 unwind label %16

15:                                               ; preds = %14
  call void @"??1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %4) #3
  ret void

16:                                               ; preds = %14, %1
  %17 = cleanuppad within none []
  call void @"??1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %4) #3 [ "funclet"(token %17) ]
  cleanupret from %17 unwind to caller
}

; Function Attrs: nounwind
declare dso_local i64 @strlen(ptr noundef) #19

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z"(ptr noundef nonnull returned align 8 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(8) %1) unnamed_addr #10 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  store ptr %1, ptr %4, align 8
  store ptr %0, ptr %5, align 8
  %7 = load ptr, ptr %5, align 8
  store ptr %7, ptr %3, align 8
  %8 = getelementptr inbounds nuw %"class.std::basic_ostream<char>::_Sentry_base", ptr %7, i32 0, i32 0
  %9 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  store ptr %9, ptr %8, align 8
  %10 = getelementptr inbounds nuw %"class.std::basic_ostream<char>::_Sentry_base", ptr %7, i32 0, i32 0
  %11 = load ptr, ptr %10, align 8, !nonnull !16, !align !17
  %12 = getelementptr inbounds i8, ptr %11, i64 0
  %13 = load ptr, ptr %12, align 8
  %14 = getelementptr inbounds i32, ptr %13, i32 1
  %15 = load i32, ptr %14, align 4
  %16 = sext i32 %15 to i64
  %17 = add nsw i64 0, %16
  %18 = getelementptr inbounds i8, ptr %11, i64 %17
  %19 = call noundef ptr @"?rdbuf@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_streambuf@DU?$char_traits@D@std@@@2@XZ"(ptr noundef nonnull align 8 dereferenceable(96) %18) #3
  store ptr %19, ptr %6, align 8
  %20 = load ptr, ptr %6, align 8
  %21 = icmp ne ptr %20, null
  br i1 %21, label %22, label %27

22:                                               ; preds = %2
  %23 = load ptr, ptr %6, align 8
  %24 = load ptr, ptr %23, align 8
  %25 = getelementptr inbounds ptr, ptr %24, i64 1
  %26 = load ptr, ptr %25, align 8
  call void %26(ptr noundef nonnull align 8 dereferenceable(104) %23)
  br label %27

27:                                               ; preds = %22, %2
  %28 = load ptr, ptr %3, align 8
  ret ptr %28
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef zeroext i1 @"?good@ios_base@std@@QEBA_NXZ"(ptr noundef nonnull align 8 dereferenceable(72) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call noundef i32 @"?rdstate@ios_base@std@@QEBAHXZ"(ptr noundef nonnull align 8 dereferenceable(72) %3) #3
  %5 = icmp eq i32 %4, 0
  ret i1 %5
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"?tie@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_ostream@DU?$char_traits@D@std@@@2@XZ"(ptr noundef nonnull align 8 dereferenceable(96) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"class.std::basic_ios", ptr %3, i32 0, i32 2
  %5 = load ptr, ptr %4, align 8
  ret ptr %5
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(8) ptr @"?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) #10 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca %"class.std::basic_ostream<char>::sentry", align 8
  %5 = alloca i32, align 4
  store ptr %0, ptr %2, align 8
  %6 = load ptr, ptr %2, align 8
  %7 = getelementptr inbounds i8, ptr %6, i64 0
  %8 = load ptr, ptr %7, align 8
  %9 = getelementptr inbounds i32, ptr %8, i32 1
  %10 = load i32, ptr %9, align 4
  %11 = sext i32 %10 to i64
  %12 = add nsw i64 0, %11
  %13 = getelementptr inbounds i8, ptr %6, i64 %12
  %14 = call noundef ptr @"?rdbuf@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_streambuf@DU?$char_traits@D@std@@@2@XZ"(ptr noundef nonnull align 8 dereferenceable(96) %13) #3
  store ptr %14, ptr %3, align 8
  %15 = load ptr, ptr %3, align 8
  %16 = icmp ne ptr %15, null
  br i1 %16, label %17, label %60

17:                                               ; preds = %1
  %18 = call noundef ptr @"??0sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z"(ptr noundef nonnull align 8 dereferenceable(16) %4, ptr noundef nonnull align 8 dereferenceable(8) %6)
  invoke void @llvm.seh.scope.begin()
          to label %19 unwind label %58

19:                                               ; preds = %17
  %20 = invoke noundef zeroext i1 @"??Bsentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEBA_NXZ"(ptr noundef nonnull align 8 dereferenceable(16) %4)
          to label %21 unwind label %58

21:                                               ; preds = %19
  br i1 %20, label %22, label %56

22:                                               ; preds = %21
  store i32 0, ptr %5, align 4
  invoke void @llvm.seh.try.begin()
          to label %23 unwind label %26

23:                                               ; preds = %22
  %24 = load ptr, ptr %3, align 8
  %25 = invoke noundef i32 @"?pubsync@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ"(ptr noundef nonnull align 8 dereferenceable(104) %24)
          to label %50 unwind label %26

26:                                               ; preds = %23, %22
  %27 = catchswitch within none [label %28] unwind label %58

28:                                               ; preds = %26
  %29 = catchpad within %27 [ptr null, i32 0, ptr null]
  %30 = getelementptr inbounds i8, ptr %6, i64 0
  %31 = load ptr, ptr %30, align 8
  %32 = getelementptr inbounds i32, ptr %31, i32 1
  %33 = load i32, ptr %32, align 4
  %34 = sext i32 %33 to i64
  %35 = add nsw i64 0, %34
  %36 = getelementptr inbounds i8, ptr %6, i64 %35
  invoke void @"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"(ptr noundef nonnull align 8 dereferenceable(96) %36, i32 noundef 4, i1 noundef zeroext true) [ "funclet"(token %29) ]
          to label %37 unwind label %58

37:                                               ; preds = %28
  invoke void @llvm.seh.scope.end() [ "funclet"(token %29) ]
          to label %38 unwind label %58

38:                                               ; preds = %37
  catchret from %29 to label %39

39:                                               ; preds = %38
  br label %40

40:                                               ; preds = %39, %55
  %41 = getelementptr inbounds i8, ptr %6, i64 0
  %42 = load ptr, ptr %41, align 8
  %43 = getelementptr inbounds i32, ptr %42, i32 1
  %44 = load i32, ptr %43, align 4
  %45 = sext i32 %44 to i64
  %46 = add nsw i64 0, %45
  %47 = getelementptr inbounds i8, ptr %6, i64 %46
  %48 = load i32, ptr %5, align 4
  invoke void @"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"(ptr noundef nonnull align 8 dereferenceable(96) %47, i32 noundef %48, i1 noundef zeroext false)
          to label %49 unwind label %58

49:                                               ; preds = %40
  br label %56

50:                                               ; preds = %23
  %51 = icmp eq i32 %25, -1
  br i1 %51, label %52, label %55

52:                                               ; preds = %50
  %53 = load i32, ptr %5, align 4
  %54 = or i32 %53, 4
  store i32 %54, ptr %5, align 4
  br label %55

55:                                               ; preds = %52, %50
  br label %40

56:                                               ; preds = %49, %21
  invoke void @llvm.seh.scope.end()
          to label %57 unwind label %58

57:                                               ; preds = %56
  call void @"??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %4) #3
  br label %60

58:                                               ; preds = %56, %40, %37, %28, %26, %19, %17
  %59 = cleanuppad within none []
  call void @"??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %4) #3 [ "funclet"(token %59) ]
  cleanupret from %59 unwind to caller

60:                                               ; preds = %57, %1
  ret ptr %6
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"??1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) unnamed_addr #0 comdat align 2 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr inbounds nuw %"class.std::basic_ostream<char>::_Sentry_base", ptr %4, i32 0, i32 0
  %6 = load ptr, ptr %5, align 8, !nonnull !16, !align !17
  %7 = getelementptr inbounds i8, ptr %6, i64 0
  %8 = load ptr, ptr %7, align 8
  %9 = getelementptr inbounds i32, ptr %8, i32 1
  %10 = load i32, ptr %9, align 4
  %11 = sext i32 %10 to i64
  %12 = add nsw i64 0, %11
  %13 = getelementptr inbounds i8, ptr %6, i64 %12
  %14 = call noundef ptr @"?rdbuf@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_streambuf@DU?$char_traits@D@std@@@2@XZ"(ptr noundef nonnull align 8 dereferenceable(96) %13) #3
  store ptr %14, ptr %3, align 8
  %15 = load ptr, ptr %3, align 8
  %16 = icmp ne ptr %15, null
  br i1 %16, label %17, label %22

17:                                               ; preds = %1
  %18 = load ptr, ptr %3, align 8
  %19 = load ptr, ptr %18, align 8
  %20 = getelementptr inbounds ptr, ptr %19, i64 2
  %21 = load ptr, ptr %20, align 8
  call void %21(ptr noundef nonnull align 8 dereferenceable(104) %18)
  br label %22

22:                                               ; preds = %17, %1
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef i32 @"?rdstate@ios_base@std@@QEBAHXZ"(ptr noundef nonnull align 8 dereferenceable(72) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"class.std::ios_base", ptr %3, i32 0, i32 2
  %5 = load i32, ptr %4, align 8
  ret i32 %5
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local noundef i32 @"?pubsync@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ"(ptr noundef nonnull align 8 dereferenceable(104) %0) #10 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = load ptr, ptr %3, align 8
  %5 = getelementptr inbounds ptr, ptr %4, i64 13
  %6 = load ptr, ptr %5, align 8
  %7 = call noundef i32 %6(ptr noundef nonnull align 8 dereferenceable(104) %3)
  ret i32 %7
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef i64 @"?_Pnavail@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEBA_JXZ"(ptr noundef nonnull align 8 dereferenceable(104) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"class.std::basic_streambuf", ptr %3, i32 0, i32 8
  %5 = load ptr, ptr %4, align 8
  %6 = load ptr, ptr %5, align 8
  %7 = icmp ne ptr %6, null
  br i1 %7, label %8, label %12

8:                                                ; preds = %1
  %9 = getelementptr inbounds nuw %"class.std::basic_streambuf", ptr %3, i32 0, i32 12
  %10 = load ptr, ptr %9, align 8
  %11 = load i32, ptr %10, align 4
  br label %13

12:                                               ; preds = %1
  br label %13

13:                                               ; preds = %12, %8
  %14 = phi i32 [ %11, %8 ], [ 0, %12 ]
  %15 = sext i32 %14 to i64
  ret i64 %15
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef i32 @"?to_int_type@?$_Narrow_char_traits@DH@std@@SAHD@Z"(i8 noundef %0) #0 comdat align 2 {
  %2 = alloca i8, align 1
  store i8 %0, ptr %2, align 1
  %3 = load i8, ptr %2, align 1
  %4 = zext i8 %3 to i32
  ret i32 %4
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"?_Pninc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAPEADXZ"(ptr noundef nonnull align 8 dereferenceable(104) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"class.std::basic_streambuf", ptr %3, i32 0, i32 12
  %5 = load ptr, ptr %4, align 8
  %6 = load i32, ptr %5, align 4
  %7 = add nsw i32 %6, -1
  store i32 %7, ptr %5, align 4
  %8 = getelementptr inbounds nuw %"class.std::basic_streambuf", ptr %3, i32 0, i32 8
  %9 = load ptr, ptr %8, align 8
  %10 = load ptr, ptr %9, align 8
  %11 = getelementptr inbounds nuw i8, ptr %10, i32 1
  store ptr %11, ptr %9, align 8
  ret ptr %10
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local void @"?clear@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"(ptr noundef nonnull align 8 dereferenceable(96) %0, i32 noundef %1, i1 noundef zeroext %2) #10 comdat align 2 {
  %4 = alloca i8, align 1
  %5 = alloca i32, align 4
  %6 = alloca ptr, align 8
  %7 = zext i1 %2 to i8
  store i8 %7, ptr %4, align 1
  store i32 %1, ptr %5, align 4
  store ptr %0, ptr %6, align 8
  %8 = load ptr, ptr %6, align 8
  %9 = load i8, ptr %4, align 1
  %10 = trunc i8 %9 to i1
  %11 = load i32, ptr %5, align 4
  %12 = getelementptr inbounds nuw %"class.std::basic_ios", ptr %8, i32 0, i32 1
  %13 = load ptr, ptr %12, align 8
  %14 = icmp ne ptr %13, null
  %15 = zext i1 %14 to i64
  %16 = select i1 %14, i32 0, i32 4
  %17 = or i32 %11, %16
  call void @"?clear@ios_base@std@@QEAAXH_N@Z"(ptr noundef nonnull align 8 dereferenceable(72) %8, i32 noundef %17, i1 noundef zeroext %10)
  ret void
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local void @"?clear@ios_base@std@@QEAAXH_N@Z"(ptr noundef nonnull align 8 dereferenceable(72) %0, i32 noundef %1, i1 noundef zeroext %2) #10 comdat align 2 {
  %4 = alloca i8, align 1
  %5 = alloca i32, align 4
  %6 = alloca ptr, align 8
  %7 = alloca i32, align 4
  %8 = alloca ptr, align 8
  %9 = alloca %"class.std::ios_base::failure", align 8
  %10 = alloca %"class.std::error_code", align 8
  %11 = zext i1 %2 to i8
  store i8 %11, ptr %4, align 1
  store i32 %1, ptr %5, align 4
  store ptr %0, ptr %6, align 8
  %12 = load ptr, ptr %6, align 8
  %13 = load i32, ptr %5, align 4
  %14 = and i32 %13, 23
  store i32 %14, ptr %5, align 4
  %15 = load i32, ptr %5, align 4
  %16 = getelementptr inbounds nuw %"class.std::ios_base", ptr %12, i32 0, i32 2
  store i32 %15, ptr %16, align 8
  %17 = load i32, ptr %5, align 4
  %18 = getelementptr inbounds nuw %"class.std::ios_base", ptr %12, i32 0, i32 3
  %19 = load i32, ptr %18, align 4
  %20 = and i32 %17, %19
  store i32 %20, ptr %7, align 4
  %21 = load i32, ptr %7, align 4
  %22 = icmp ne i32 %21, 0
  br i1 %22, label %23, label %42

23:                                               ; preds = %3
  %24 = load i8, ptr %4, align 1
  %25 = trunc i8 %24 to i1
  br i1 %25, label %26, label %27

26:                                               ; preds = %23
  call void @_CxxThrowException(ptr null, ptr null) #22
  unreachable

27:                                               ; preds = %23
  %28 = load i32, ptr %7, align 4
  %29 = and i32 %28, 4
  %30 = icmp ne i32 %29, 0
  br i1 %30, label %31, label %32

31:                                               ; preds = %27
  store ptr @"??_C@_0BF@PHHKMMFD@ios_base?3?3badbit?5set?$AA@", ptr %8, align 8
  br label %39

32:                                               ; preds = %27
  %33 = load i32, ptr %7, align 4
  %34 = and i32 %33, 2
  %35 = icmp ne i32 %34, 0
  br i1 %35, label %36, label %37

36:                                               ; preds = %32
  store ptr @"??_C@_0BG@FMKFHCIL@ios_base?3?3failbit?5set?$AA@", ptr %8, align 8
  br label %38

37:                                               ; preds = %32
  store ptr @"??_C@_0BF@OOHOMBOF@ios_base?3?3eofbit?5set?$AA@", ptr %8, align 8
  br label %38

38:                                               ; preds = %37, %36
  br label %39

39:                                               ; preds = %38, %31
  call void @"?make_error_code@std@@YA?AVerror_code@1@W4io_errc@1@@Z"(ptr dead_on_unwind writable sret(%"class.std::error_code") align 8 %10, i32 noundef 1) #3
  %40 = load ptr, ptr %8, align 8
  %41 = call noundef ptr @"??0failure@ios_base@std@@QEAA@PEBDAEBVerror_code@2@@Z"(ptr noundef nonnull align 8 dereferenceable(40) %9, ptr noundef %40, ptr noundef nonnull align 8 dereferenceable(16) %10)
  call void @_CxxThrowException(ptr %9, ptr @"_TI5?AVfailure@ios_base@std@@") #22
  unreachable

42:                                               ; preds = %3
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?make_error_code@std@@YA?AVerror_code@1@W4io_errc@1@@Z"(ptr dead_on_unwind noalias writable sret(%"class.std::error_code") align 8 %0, i32 noundef %1) #0 comdat {
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  store ptr %0, ptr %3, align 8
  store i32 %1, ptr %4, align 4
  %5 = call noundef nonnull align 8 dereferenceable(16) ptr @"?iostream_category@std@@YAAEBVerror_category@1@XZ"() #3
  %6 = load i32, ptr %4, align 4
  %7 = call noundef ptr @"??0error_code@std@@QEAA@HAEBVerror_category@1@@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, i32 noundef %6, ptr noundef nonnull align 8 dereferenceable(16) %5) #3
  ret void
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0failure@ios_base@std@@QEAA@PEBDAEBVerror_code@2@@Z"(ptr noundef nonnull returned align 8 dereferenceable(40) %0, ptr noundef %1, ptr noundef nonnull align 8 dereferenceable(16) %2) unnamed_addr #10 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca %"class.std::error_code", align 8
  store ptr %2, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store ptr %0, ptr %6, align 8
  %8 = load ptr, ptr %6, align 8
  %9 = load ptr, ptr %5, align 8
  %10 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %7, ptr align 8 %10, i64 16, i1 false)
  %11 = call noundef ptr @"??0system_error@std@@QEAA@Verror_code@1@PEBD@Z"(ptr noundef nonnull align 8 dereferenceable(40) %8, ptr noundef %7, ptr noundef %9)
  invoke void @llvm.seh.scope.begin()
          to label %12 unwind label %14

12:                                               ; preds = %3
  store ptr @"??_7failure@ios_base@std@@6B@", ptr %8, align 8
  invoke void @llvm.seh.scope.end()
          to label %13 unwind label %14

13:                                               ; preds = %12
  ret ptr %8

14:                                               ; preds = %12, %3
  %15 = cleanuppad within none []
  call void @"??1system_error@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(40) %8) #3 [ "funclet"(token %15) ]
  cleanupret from %15 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0failure@ios_base@std@@QEAA@AEBV012@@Z"(ptr noundef nonnull returned align 8 dereferenceable(40) %0, ptr noundef nonnull align 8 dereferenceable(40) %1) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %7 = call noundef ptr @"??0system_error@std@@QEAA@AEBV01@@Z"(ptr noundef nonnull align 8 dereferenceable(40) %5, ptr noundef nonnull align 8 dereferenceable(40) %6) #3
  invoke void @llvm.seh.scope.begin()
          to label %8 unwind label %10

8:                                                ; preds = %2
  store ptr @"??_7failure@ios_base@std@@6B@", ptr %5, align 8
  invoke void @llvm.seh.scope.end()
          to label %9 unwind label %10

9:                                                ; preds = %8
  ret ptr %5

10:                                               ; preds = %8, %2
  %11 = cleanuppad within none []
  call void @"??1system_error@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(40) %5) #3 [ "funclet"(token %11) ]
  cleanupret from %11 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0system_error@std@@QEAA@AEBV01@@Z"(ptr noundef nonnull returned align 8 dereferenceable(40) %0, ptr noundef nonnull align 8 dereferenceable(40) %1) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %7 = call noundef ptr @"??0_System_error@std@@QEAA@AEBV01@@Z"(ptr noundef nonnull align 8 dereferenceable(40) %5, ptr noundef nonnull align 8 dereferenceable(40) %6) #3
  invoke void @llvm.seh.scope.begin()
          to label %8 unwind label %10

8:                                                ; preds = %2
  store ptr @"??_7system_error@std@@6B@", ptr %5, align 8
  invoke void @llvm.seh.scope.end()
          to label %9 unwind label %10

9:                                                ; preds = %8
  ret ptr %5

10:                                               ; preds = %8, %2
  %11 = cleanuppad within none []
  call void @"??1_System_error@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(40) %5) #3 [ "funclet"(token %11) ]
  cleanupret from %11 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0_System_error@std@@QEAA@AEBV01@@Z"(ptr noundef nonnull returned align 8 dereferenceable(40) %0, ptr noundef nonnull align 8 dereferenceable(40) %1) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %7 = call noundef ptr @"??0runtime_error@std@@QEAA@AEBV01@@Z"(ptr noundef nonnull align 8 dereferenceable(24) %5, ptr noundef nonnull align 8 dereferenceable(24) %6) #3
  invoke void @llvm.seh.scope.begin()
          to label %8 unwind label %13

8:                                                ; preds = %2
  store ptr @"??_7_System_error@std@@6B@", ptr %5, align 8
  %9 = getelementptr inbounds nuw %"class.std::_System_error", ptr %5, i32 0, i32 1
  %10 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %11 = getelementptr inbounds nuw %"class.std::_System_error", ptr %10, i32 0, i32 1
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %9, ptr align 8 %11, i64 16, i1 false)
  invoke void @llvm.seh.scope.end()
          to label %12 unwind label %13

12:                                               ; preds = %8
  ret ptr %5

13:                                               ; preds = %8, %2
  %14 = cleanuppad within none []
  call void @"??1runtime_error@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %5) #3 [ "funclet"(token %14) ]
  cleanupret from %14 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"??1failure@ios_base@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(40) %0) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  invoke void @llvm.seh.scope.begin()
          to label %4 unwind label %6

4:                                                ; preds = %1
  invoke void @llvm.seh.scope.end()
          to label %5 unwind label %6

5:                                                ; preds = %4
  call void @"??1system_error@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(40) %3) #3
  ret void

6:                                                ; preds = %4, %1
  %7 = cleanuppad within none []
  call void @"??1system_error@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(40) %3) #3 [ "funclet"(token %7) ]
  cleanupret from %7 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(16) ptr @"?iostream_category@std@@YAAEBVerror_category@1@XZ"() #0 comdat {
  %1 = call noundef nonnull align 8 dereferenceable(16) ptr @"??$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@0@XZ"() #3
  ret ptr %1
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0error_code@std@@QEAA@HAEBVerror_category@1@@Z"(ptr noundef nonnull returned align 8 dereferenceable(16) %0, i32 noundef %1, ptr noundef nonnull align 8 dereferenceable(16) %2) unnamed_addr #0 comdat align 2 {
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  %6 = alloca ptr, align 8
  store ptr %2, ptr %4, align 8
  store i32 %1, ptr %5, align 4
  store ptr %0, ptr %6, align 8
  %7 = load ptr, ptr %6, align 8
  %8 = getelementptr inbounds nuw %"class.std::error_code", ptr %7, i32 0, i32 0
  %9 = load i32, ptr %5, align 4
  store i32 %9, ptr %8, align 8
  %10 = getelementptr inbounds nuw %"class.std::error_code", ptr %7, i32 0, i32 1
  %11 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  store ptr %11, ptr %10, align 8
  ret ptr %7
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(16) ptr @"??$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@0@XZ"() #0 comdat {
  ret ptr @"?_Static@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@1@XZ@4V21@B"
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??_G_Iostream_error_category2@std@@UEAAPEAXI@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, i32 noundef %1) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  %5 = alloca ptr, align 8
  store i32 %1, ptr %4, align 4
  store ptr %0, ptr %5, align 8
  %6 = load ptr, ptr %5, align 8
  store ptr %6, ptr %3, align 8
  %7 = load i32, ptr %4, align 4
  invoke void @llvm.seh.scope.begin()
          to label %8 unwind label %14

8:                                                ; preds = %2
  call void @"??1_Iostream_error_category2@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %6) #3
  invoke void @llvm.seh.scope.end()
          to label %9 unwind label %14

9:                                                ; preds = %8
  %10 = icmp eq i32 %7, 0
  br i1 %10, label %12, label %11

11:                                               ; preds = %9
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %6, i64 noundef 16) #23
  br label %12

12:                                               ; preds = %11, %9
  %13 = load ptr, ptr %3, align 8
  ret ptr %13

14:                                               ; preds = %8, %2
  %15 = cleanuppad within none []
  %16 = icmp eq i32 %7, 0
  br i1 %16, label %18, label %17

17:                                               ; preds = %14
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %6, i64 noundef 16) #23 [ "funclet"(token %15) ]
  br label %18

18:                                               ; preds = %17, %14
  cleanupret from %15 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"?name@_Iostream_error_category2@std@@UEBAPEBDXZ"(ptr noundef nonnull align 8 dereferenceable(16) %0) unnamed_addr #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret ptr @"??_C@_08LLGCOLLL@iostream?$AA@"
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local void @"?message@_Iostream_error_category2@std@@UEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@H@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr dead_on_unwind noalias writable sret(%"class.std::basic_string") align 8 %1, i32 noundef %2) unnamed_addr #10 comdat align 2 {
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  %6 = alloca ptr, align 8
  %7 = alloca i64, align 8
  store ptr %1, ptr %4, align 8
  store i32 %2, ptr %5, align 4
  store ptr %0, ptr %6, align 8
  %8 = load ptr, ptr %6, align 8
  %9 = load i32, ptr %5, align 4
  %10 = icmp eq i32 %9, 1
  br i1 %10, label %11, label %13

11:                                               ; preds = %3
  store i64 21, ptr %7, align 8
  %12 = call noundef ptr @"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD_K@Z"(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef @"?_Iostream_error@?4??message@_Iostream_error_category2@std@@UEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@3@H@Z@4QBDB", i64 noundef 21)
  br label %17

13:                                               ; preds = %3
  %14 = load i32, ptr %5, align 4
  %15 = call noundef ptr @"?_Syserror_map@std@@YAPEBDH@Z"(i32 noundef %14)
  %16 = call noundef ptr @"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD@Z"(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef %15)
  br label %17

17:                                               ; preds = %13, %11
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?default_error_condition@error_category@std@@UEBA?AVerror_condition@2@H@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr dead_on_unwind noalias writable sret(%"class.std::error_condition") align 8 %1, i32 noundef %2) unnamed_addr #0 comdat align 2 {
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  %6 = alloca ptr, align 8
  store ptr %1, ptr %4, align 8
  store i32 %2, ptr %5, align 4
  store ptr %0, ptr %6, align 8
  %7 = load ptr, ptr %6, align 8
  %8 = load i32, ptr %5, align 4
  %9 = call noundef ptr @"??0error_condition@std@@QEAA@HAEBVerror_category@1@@Z"(ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %8, ptr noundef nonnull align 8 dereferenceable(16) %7) #3
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef zeroext i1 @"?equivalent@error_category@std@@UEBA_NAEBVerror_code@2@H@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) unnamed_addr #0 comdat align 2 {
  %4 = alloca i32, align 4
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  store i32 %2, ptr %4, align 4
  store ptr %1, ptr %5, align 8
  store ptr %0, ptr %6, align 8
  %7 = load ptr, ptr %6, align 8
  %8 = load ptr, ptr %5, align 8, !nonnull !16, !align !17
  %9 = call noundef nonnull align 8 dereferenceable(16) ptr @"?category@error_code@std@@QEBAAEBVerror_category@2@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %8) #3
  %10 = call noundef zeroext i1 @"??8error_category@std@@QEBA_NAEBV01@@Z"(ptr noundef nonnull align 8 dereferenceable(16) %7, ptr noundef nonnull align 8 dereferenceable(16) %9) #3
  br i1 %10, label %11, label %16

11:                                               ; preds = %3
  %12 = load ptr, ptr %5, align 8, !nonnull !16, !align !17
  %13 = call noundef i32 @"?value@error_code@std@@QEBAHXZ"(ptr noundef nonnull align 8 dereferenceable(16) %12) #3
  %14 = load i32, ptr %4, align 4
  %15 = icmp eq i32 %13, %14
  br label %16

16:                                               ; preds = %11, %3
  %17 = phi i1 [ false, %3 ], [ %15, %11 ]
  ret i1 %17
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef zeroext i1 @"?equivalent@error_category@std@@UEBA_NHAEBVerror_condition@2@@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, i32 noundef %1, ptr noundef nonnull align 8 dereferenceable(16) %2) unnamed_addr #0 comdat align 2 {
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  %6 = alloca ptr, align 8
  %7 = alloca %"class.std::error_condition", align 8
  store ptr %2, ptr %4, align 8
  store i32 %1, ptr %5, align 4
  store ptr %0, ptr %6, align 8
  %8 = load ptr, ptr %6, align 8
  %9 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %10 = load i32, ptr %5, align 4
  %11 = load ptr, ptr %8, align 8
  %12 = getelementptr inbounds ptr, ptr %11, i64 3
  %13 = load ptr, ptr %12, align 8
  call void %13(ptr noundef nonnull align 8 dereferenceable(16) %8, ptr dead_on_unwind writable sret(%"class.std::error_condition") align 8 %7, i32 noundef %10) #3
  %14 = call noundef zeroext i1 @"??8std@@YA_NAEBVerror_condition@0@0@Z"(ptr noundef nonnull align 8 dereferenceable(16) %7, ptr noundef nonnull align 8 dereferenceable(16) %9) #3
  ret i1 %14
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"??1_Iostream_error_category2@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %0) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  invoke void @llvm.seh.scope.begin()
          to label %4 unwind label %6

4:                                                ; preds = %1
  invoke void @llvm.seh.scope.end()
          to label %5 unwind label %6

5:                                                ; preds = %4
  call void @"??1error_category@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %3) #3
  ret void

6:                                                ; preds = %4, %1
  %7 = cleanuppad within none []
  call void @"??1error_category@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %3) #3 [ "funclet"(token %7) ]
  cleanupret from %7 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"??1error_category@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %0) unnamed_addr #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret void
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD_K@Z"(ptr noundef nonnull returned align 8 dereferenceable(32) %0, ptr noundef %1, i64 noundef %2) unnamed_addr #10 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %4 = alloca i64, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca %"struct.std::_Zero_then_variadic_args_t", align 1
  store i64 %2, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store ptr %0, ptr %6, align 8
  %8 = load ptr, ptr %6, align 8
  %9 = getelementptr inbounds nuw %"class.std::basic_string", ptr %8, i32 0, i32 0
  %10 = getelementptr inbounds nuw %"struct.std::_Zero_then_variadic_args_t", ptr %7, i32 0, i32 0
  %11 = load i8, ptr %10, align 1
  %12 = call noundef ptr @"??$?0$$V@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@U_Zero_then_variadic_args_t@1@@Z"(ptr noundef nonnull align 8 dereferenceable(32) %9, i8 %11) #3
  invoke void @llvm.seh.scope.begin()
          to label %13 unwind label %18

13:                                               ; preds = %3
  %14 = load i64, ptr %4, align 8
  %15 = load ptr, ptr %5, align 8
  invoke void @"??$_Construct@$00PEBD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXQEBD_K@Z"(ptr noundef nonnull align 8 dereferenceable(32) %8, ptr noundef %15, i64 noundef %14)
          to label %16 unwind label %18

16:                                               ; preds = %13
  invoke void @llvm.seh.scope.end()
          to label %17 unwind label %18

17:                                               ; preds = %16
  ret ptr %8

18:                                               ; preds = %16, %13, %3
  %19 = cleanuppad within none []
  call void @"??1?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %9) #3 [ "funclet"(token %19) ]
  cleanupret from %19 unwind to caller
}

declare dso_local noundef ptr @"?_Syserror_map@std@@YAPEBDH@Z"(i32 noundef) #14

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD@Z"(ptr noundef nonnull returned align 8 dereferenceable(32) %0, ptr noundef %1) unnamed_addr #10 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca %"struct.std::_Zero_then_variadic_args_t", align 1
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = getelementptr inbounds nuw %"class.std::basic_string", ptr %6, i32 0, i32 0
  %8 = getelementptr inbounds nuw %"struct.std::_Zero_then_variadic_args_t", ptr %5, i32 0, i32 0
  %9 = load i8, ptr %8, align 1
  %10 = call noundef ptr @"??$?0$$V@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@U_Zero_then_variadic_args_t@1@@Z"(ptr noundef nonnull align 8 dereferenceable(32) %7, i8 %9) #3
  invoke void @llvm.seh.scope.begin()
          to label %11 unwind label %18

11:                                               ; preds = %2
  %12 = load ptr, ptr %3, align 8
  %13 = call noundef i64 @"?length@?$_Narrow_char_traits@DH@std@@SA_KQEBD@Z"(ptr noundef %12) #3
  %14 = call noundef i64 @"??$_Convert_size@_K_K@std@@YA_K_K@Z"(i64 noundef %13) #3
  %15 = load ptr, ptr %3, align 8
  invoke void @"??$_Construct@$00PEBD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXQEBD_K@Z"(ptr noundef nonnull align 8 dereferenceable(32) %6, ptr noundef %15, i64 noundef %14)
          to label %16 unwind label %18

16:                                               ; preds = %11
  invoke void @llvm.seh.scope.end()
          to label %17 unwind label %18

17:                                               ; preds = %16
  ret ptr %6

18:                                               ; preds = %16, %11, %2
  %19 = cleanuppad within none []
  call void @"??1?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %7) #3 [ "funclet"(token %19) ]
  cleanupret from %19 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??$?0$$V@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@U_Zero_then_variadic_args_t@1@@Z"(ptr noundef nonnull returned align 8 dereferenceable(32) %0, i8 %1) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %3 = alloca %"struct.std::_Zero_then_variadic_args_t", align 1
  %4 = alloca ptr, align 8
  %5 = getelementptr inbounds nuw %"struct.std::_Zero_then_variadic_args_t", ptr %3, i32 0, i32 0
  store i8 %1, ptr %5, align 1
  store ptr %0, ptr %4, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = call noundef ptr @"??0?$allocator@D@std@@QEAA@XZ"(ptr noundef nonnull align 1 dereferenceable(1) %6) #3
  %8 = getelementptr inbounds nuw %"class.std::_Compressed_pair.10", ptr %6, i32 0, i32 0
  %9 = call noundef ptr @"??0?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %8) #3
  invoke void @llvm.seh.scope.begin()
          to label %10 unwind label %12

10:                                               ; preds = %2
  invoke void @llvm.seh.scope.end()
          to label %11 unwind label %12

11:                                               ; preds = %10
  ret ptr %6

12:                                               ; preds = %10, %2
  %13 = cleanuppad within none []
  call void @"??1?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %8) #3 [ "funclet"(token %13) ]
  cleanupret from %13 unwind to caller
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local void @"??$_Construct@$00PEBD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXQEBD_K@Z"(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef %1, i64 noundef %2) #10 comdat align 2 {
  %4 = alloca i64, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca %"struct.std::_Fake_proxy_ptr_impl", align 1
  %11 = alloca i8, align 1
  %12 = alloca i64, align 8
  %13 = alloca ptr, align 8
  %14 = alloca i8, align 1
  store i64 %2, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store ptr %0, ptr %6, align 8
  %15 = load ptr, ptr %6, align 8
  %16 = getelementptr inbounds nuw %"class.std::basic_string", ptr %15, i32 0, i32 0
  %17 = getelementptr inbounds nuw %"class.std::_Compressed_pair.10", ptr %16, i32 0, i32 0
  store ptr %17, ptr %7, align 8
  %18 = load i64, ptr %4, align 8
  %19 = call noundef i64 @"?max_size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ"(ptr noundef nonnull align 8 dereferenceable(32) %15) #3
  %20 = icmp ugt i64 %18, %19
  br i1 %20, label %21, label %22

21:                                               ; preds = %3
  call void @"?_Xlen_string@std@@YAXXZ"() #22
  unreachable

22:                                               ; preds = %3
  %23 = call noundef nonnull align 1 dereferenceable(1) ptr @"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV?$allocator@D@2@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %15) #3
  store ptr %23, ptr %8, align 8
  store ptr @"?_Fake_alloc@std@@3U_Fake_allocator@1@B", ptr %9, align 8
  %24 = load ptr, ptr %7, align 8, !nonnull !16, !align !17
  %25 = call noundef ptr @"??0_Fake_proxy_ptr_impl@std@@QEAA@AEBU_Fake_allocator@1@AEBU_Container_base0@1@@Z"(ptr noundef nonnull align 1 dereferenceable(1) %10, ptr noundef nonnull align 1 dereferenceable(1) @"?_Fake_alloc@std@@3U_Fake_allocator@1@B", ptr noundef nonnull align 1 dereferenceable(1) %24) #3
  %26 = load i64, ptr %4, align 8
  %27 = icmp ule i64 %26, 15
  br i1 %27, label %28, label %44

28:                                               ; preds = %22
  %29 = load i64, ptr %4, align 8
  %30 = load ptr, ptr %7, align 8, !nonnull !16, !align !17
  %31 = getelementptr inbounds nuw %"class.std::_String_val", ptr %30, i32 0, i32 1
  store i64 %29, ptr %31, align 8
  %32 = load ptr, ptr %7, align 8, !nonnull !16, !align !17
  %33 = getelementptr inbounds nuw %"class.std::_String_val", ptr %32, i32 0, i32 2
  store i64 15, ptr %33, align 8
  %34 = load i64, ptr %4, align 8
  %35 = load ptr, ptr %5, align 8
  %36 = load ptr, ptr %7, align 8, !nonnull !16, !align !17
  %37 = getelementptr inbounds nuw %"class.std::_String_val", ptr %36, i32 0, i32 0
  %38 = getelementptr inbounds [16 x i8], ptr %37, i64 0, i64 0
  %39 = call noundef ptr @"?copy@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z"(ptr noundef %38, ptr noundef %35, i64 noundef %34) #3
  store i8 0, ptr %11, align 1
  %40 = load ptr, ptr %7, align 8, !nonnull !16, !align !17
  %41 = getelementptr inbounds nuw %"class.std::_String_val", ptr %40, i32 0, i32 0
  %42 = load i64, ptr %4, align 8
  %43 = getelementptr inbounds nuw [16 x i8], ptr %41, i64 0, i64 %42
  call void @"?assign@?$_Narrow_char_traits@DH@std@@SAXAEADAEBD@Z"(ptr noundef nonnull align 1 dereferenceable(1) %43, ptr noundef nonnull align 1 dereferenceable(1) %11) #3
  call void @"?_Release@_Fake_proxy_ptr_impl@std@@QEAAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %10) #3
  br label %68

44:                                               ; preds = %22
  %45 = load ptr, ptr %7, align 8, !nonnull !16, !align !17
  %46 = getelementptr inbounds nuw %"class.std::_String_val", ptr %45, i32 0, i32 2
  store i64 15, ptr %46, align 8
  %47 = load i64, ptr %4, align 8
  %48 = call noundef i64 @"?_Calculate_growth@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBA_K_K@Z"(ptr noundef nonnull align 8 dereferenceable(32) %15, i64 noundef %47) #3
  store i64 %48, ptr %12, align 8
  %49 = load ptr, ptr %8, align 8, !nonnull !16
  %50 = call noundef ptr @"??$_Allocate_for_capacity@$0A@@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@CAPEADAEAV?$allocator@D@1@AEA_K@Z"(ptr noundef nonnull align 1 dereferenceable(1) %49, ptr noundef nonnull align 8 dereferenceable(8) %12)
  store ptr %50, ptr %13, align 8
  %51 = load ptr, ptr %7, align 8, !nonnull !16, !align !17
  %52 = getelementptr inbounds nuw %"class.std::_String_val", ptr %51, i32 0, i32 0
  call void @"??$_Construct_in_place@PEADAEBQEAD@std@@YAXAEAPEADAEBQEAD@Z"(ptr noundef nonnull align 8 dereferenceable(8) %52, ptr noundef nonnull align 8 dereferenceable(8) %13) #3
  %53 = load i64, ptr %4, align 8
  %54 = load ptr, ptr %7, align 8, !nonnull !16, !align !17
  %55 = getelementptr inbounds nuw %"class.std::_String_val", ptr %54, i32 0, i32 1
  store i64 %53, ptr %55, align 8
  %56 = load i64, ptr %12, align 8
  %57 = load ptr, ptr %7, align 8, !nonnull !16, !align !17
  %58 = getelementptr inbounds nuw %"class.std::_String_val", ptr %57, i32 0, i32 2
  store i64 %56, ptr %58, align 8
  %59 = load i64, ptr %4, align 8
  %60 = load ptr, ptr %5, align 8
  %61 = load ptr, ptr %13, align 8
  %62 = call noundef ptr @"??$_Unfancy@D@std@@YAPEADPEAD@Z"(ptr noundef %61) #3
  %63 = call noundef ptr @"?copy@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z"(ptr noundef %62, ptr noundef %60, i64 noundef %59) #3
  store i8 0, ptr %14, align 1
  %64 = load ptr, ptr %13, align 8
  %65 = call noundef ptr @"??$_Unfancy@D@std@@YAPEADPEAD@Z"(ptr noundef %64) #3
  %66 = load i64, ptr %4, align 8
  %67 = getelementptr inbounds nuw i8, ptr %65, i64 %66
  call void @"?assign@?$_Narrow_char_traits@DH@std@@SAXAEADAEBD@Z"(ptr noundef nonnull align 1 dereferenceable(1) %67, ptr noundef nonnull align 1 dereferenceable(1) %14) #3
  call void @"?_Release@_Fake_proxy_ptr_impl@std@@QEAAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %10) #3
  br label %68

68:                                               ; preds = %44, %28
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"??1?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %0) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  invoke void @llvm.seh.scope.begin()
          to label %4 unwind label %7

4:                                                ; preds = %1
  invoke void @llvm.seh.scope.end()
          to label %5 unwind label %7

5:                                                ; preds = %4
  %6 = getelementptr inbounds nuw %"class.std::_Compressed_pair.10", ptr %3, i32 0, i32 0
  call void @"??1?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %6) #3
  ret void

7:                                                ; preds = %4, %1
  %8 = cleanuppad within none []
  %9 = getelementptr inbounds nuw %"class.std::_Compressed_pair.10", ptr %3, i32 0, i32 0
  call void @"??1?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %9) #3 [ "funclet"(token %8) ]
  cleanupret from %8 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0?$allocator@D@std@@QEAA@XZ"(ptr noundef nonnull returned align 1 dereferenceable(1) %0) unnamed_addr #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret ptr %3
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull returned align 8 dereferenceable(32) %0) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"class.std::_String_val", ptr %3, i32 0, i32 0
  %5 = call noundef ptr @"??0_Bxty@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %4) #3
  invoke void @llvm.seh.scope.begin()
          to label %6 unwind label %10

6:                                                ; preds = %1
  %7 = getelementptr inbounds nuw %"class.std::_String_val", ptr %3, i32 0, i32 1
  store i64 0, ptr %7, align 8
  %8 = getelementptr inbounds nuw %"class.std::_String_val", ptr %3, i32 0, i32 2
  store i64 0, ptr %8, align 8
  invoke void @llvm.seh.scope.end()
          to label %9 unwind label %10

9:                                                ; preds = %6
  ret ptr %3

10:                                               ; preds = %6, %1
  %11 = cleanuppad within none []
  call void @"??1_Bxty@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %4) #3 [ "funclet"(token %11) ]
  cleanupret from %11 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"??1?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %0) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  invoke void @llvm.seh.scope.begin()
          to label %4 unwind label %7

4:                                                ; preds = %1
  invoke void @llvm.seh.scope.end()
          to label %5 unwind label %7

5:                                                ; preds = %4
  %6 = getelementptr inbounds nuw %"class.std::_String_val", ptr %3, i32 0, i32 0
  call void @"??1_Bxty@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %6) #3
  ret void

7:                                                ; preds = %4, %1
  %8 = cleanuppad within none []
  %9 = getelementptr inbounds nuw %"class.std::_String_val", ptr %3, i32 0, i32 0
  call void @"??1_Bxty@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %9) #3 [ "funclet"(token %8) ]
  cleanupret from %8 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0_Bxty@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull returned align 8 dereferenceable(16) %0) unnamed_addr #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  call void @llvm.memset.p0.i64(ptr align 8 %3, i8 0, i64 16, i1 false)
  ret ptr %3
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"??1_Bxty@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %0) unnamed_addr #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef i64 @"?max_size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ"(ptr noundef nonnull align 8 dereferenceable(32) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  %3 = alloca i64, align 8
  %4 = alloca i64, align 8
  %5 = alloca i64, align 8
  %6 = alloca i64, align 8
  %7 = alloca i64, align 8
  store ptr %0, ptr %2, align 8
  %8 = load ptr, ptr %2, align 8
  %9 = call noundef nonnull align 1 dereferenceable(1) ptr @"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBAAEBV?$allocator@D@2@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %8) #3
  %10 = call noundef i64 @"?max_size@?$_Default_allocator_traits@V?$allocator@D@std@@@std@@SA_KAEBV?$allocator@D@2@@Z"(ptr noundef nonnull align 1 dereferenceable(1) %9) #3
  store i64 %10, ptr %3, align 8
  store i64 16, ptr %5, align 8
  %11 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$max@_K@std@@YAAEB_KAEB_K0@Z"(ptr noundef nonnull align 8 dereferenceable(8) %3, ptr noundef nonnull align 8 dereferenceable(8) %5) #3
  %12 = load i64, ptr %11, align 8
  store i64 %12, ptr %4, align 8
  %13 = load i64, ptr %4, align 8
  %14 = sub i64 %13, 1
  store i64 %14, ptr %6, align 8
  %15 = call noundef i64 @"?max@?$numeric_limits@_J@std@@SA_JXZ"() #3
  store i64 %15, ptr %7, align 8
  %16 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$min@_K@std@@YAAEB_KAEB_K0@Z"(ptr noundef nonnull align 8 dereferenceable(8) %7, ptr noundef nonnull align 8 dereferenceable(8) %6) #3
  %17 = load i64, ptr %16, align 8
  ret i64 %17
}

; Function Attrs: mustprogress noinline noreturn optnone sspstrong uwtable
define linkonce_odr dso_local void @"?_Xlen_string@std@@YAXXZ"() #20 comdat {
  call void @"?_Xlength_error@std@@YAXPEBD@Z"(ptr noundef @"??_C@_0BA@JFNIOLAK@string?5too?5long?$AA@") #22
  unreachable
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef nonnull align 1 dereferenceable(1) ptr @"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV?$allocator@D@2@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"class.std::basic_string", ptr %3, i32 0, i32 0
  %5 = call noundef nonnull align 1 dereferenceable(1) ptr @"?_Get_first@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAAAEAV?$allocator@D@2@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %4) #3
  ret ptr %5
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0_Fake_proxy_ptr_impl@std@@QEAA@AEBU_Fake_allocator@1@AEBU_Container_base0@1@@Z"(ptr noundef nonnull returned align 1 dereferenceable(1) %0, ptr noundef nonnull align 1 dereferenceable(1) %1, ptr noundef nonnull align 1 dereferenceable(1) %2) unnamed_addr #0 comdat align 2 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  store ptr %2, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store ptr %0, ptr %6, align 8
  %7 = load ptr, ptr %6, align 8
  ret ptr %7
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"?copy@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z"(ptr noundef %0, ptr noundef %1, i64 noundef %2) #0 comdat align 2 {
  %4 = alloca i64, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  store i64 %2, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store ptr %0, ptr %6, align 8
  %7 = load ptr, ptr %6, align 8
  %8 = load ptr, ptr %5, align 8
  %9 = load i64, ptr %4, align 8
  %10 = mul i64 %9, 1
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %7, ptr align 1 %8, i64 %10, i1 false)
  %11 = load ptr, ptr %6, align 8
  ret ptr %11
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?assign@?$_Narrow_char_traits@DH@std@@SAXAEADAEBD@Z"(ptr noundef nonnull align 1 dereferenceable(1) %0, ptr noundef nonnull align 1 dereferenceable(1) %1) #0 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %3, align 8, !nonnull !16
  %6 = load i8, ptr %5, align 1
  %7 = load ptr, ptr %4, align 8, !nonnull !16
  store i8 %6, ptr %7, align 1
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?_Release@_Fake_proxy_ptr_impl@std@@QEAAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef i64 @"?_Calculate_growth@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBA_K_K@Z"(ptr noundef nonnull align 8 dereferenceable(32) %0, i64 noundef %1) #0 comdat align 2 {
  %3 = alloca i64, align 8
  %4 = alloca ptr, align 8
  store i64 %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = call noundef i64 @"?max_size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ"(ptr noundef nonnull align 8 dereferenceable(32) %5) #3
  %7 = getelementptr inbounds nuw %"class.std::basic_string", ptr %5, i32 0, i32 0
  %8 = getelementptr inbounds nuw %"class.std::_Compressed_pair.10", ptr %7, i32 0, i32 0
  %9 = getelementptr inbounds nuw %"class.std::_String_val", ptr %8, i32 0, i32 2
  %10 = load i64, ptr %9, align 8
  %11 = load i64, ptr %3, align 8
  %12 = call noundef i64 @"?_Calculate_growth@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@CA_K_K00@Z"(i64 noundef %11, i64 noundef %10, i64 noundef %6) #3
  ret i64 %12
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??$_Allocate_for_capacity@$0A@@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@CAPEADAEAV?$allocator@D@1@AEA_K@Z"(ptr noundef nonnull align 1 dereferenceable(1) %0, ptr noundef nonnull align 8 dereferenceable(8) %1) #10 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %6 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %7 = load i64, ptr %6, align 8
  %8 = add i64 %7, 1
  store i64 %8, ptr %6, align 8
  %9 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %10 = load ptr, ptr %4, align 8, !nonnull !16
  %11 = call noundef ptr @"??$_Allocate_at_least_helper@V?$allocator@D@std@@@std@@YAPEADAEAV?$allocator@D@0@AEA_K@Z"(ptr noundef nonnull align 1 dereferenceable(1) %10, ptr noundef nonnull align 8 dereferenceable(8) %9)
  store ptr %11, ptr %5, align 8
  %12 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %13 = load i64, ptr %12, align 8
  %14 = add i64 %13, -1
  store i64 %14, ptr %12, align 8
  %15 = load ptr, ptr %5, align 8
  ret ptr %15
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"??$_Construct_in_place@PEADAEBQEAD@std@@YAXAEAPEADAEBQEAD@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(8) %1) #0 comdat {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %6 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %7 = load ptr, ptr %6, align 8
  store ptr %7, ptr %5, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??$_Unfancy@D@std@@YAPEADPEAD@Z"(ptr noundef %0) #0 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret ptr %3
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef i64 @"?max_size@?$_Default_allocator_traits@V?$allocator@D@std@@@std@@SA_KAEBV?$allocator@D@2@@Z"(ptr noundef nonnull align 1 dereferenceable(1) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  ret i64 -1
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef nonnull align 1 dereferenceable(1) ptr @"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBAAEBV?$allocator@D@2@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"class.std::basic_string", ptr %3, i32 0, i32 0
  %5 = call noundef nonnull align 1 dereferenceable(1) ptr @"?_Get_first@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEBAAEBV?$allocator@D@2@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %4) #3
  ret ptr %5
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(8) ptr @"??$max@_K@std@@YAAEB_KAEB_K0@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(8) %1) #0 comdat {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %6 = load i64, ptr %5, align 8
  %7 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %8 = load i64, ptr %7, align 8
  %9 = icmp ult i64 %6, %8
  br i1 %9, label %10, label %12

10:                                               ; preds = %2
  %11 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  br label %14

12:                                               ; preds = %2
  %13 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  br label %14

14:                                               ; preds = %12, %10
  %15 = phi ptr [ %11, %10 ], [ %13, %12 ]
  ret ptr %15
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(8) ptr @"??$min@_K@std@@YAAEB_KAEB_K0@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(8) %1) #0 comdat {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %6 = load i64, ptr %5, align 8
  %7 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %8 = load i64, ptr %7, align 8
  %9 = icmp ult i64 %6, %8
  br i1 %9, label %10, label %12

10:                                               ; preds = %2
  %11 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  br label %14

12:                                               ; preds = %2
  %13 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  br label %14

14:                                               ; preds = %12, %10
  %15 = phi ptr [ %11, %10 ], [ %13, %12 ]
  ret ptr %15
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef i64 @"?max@?$numeric_limits@_J@std@@SA_JXZ"() #0 comdat align 2 {
  ret i64 9223372036854775807
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef nonnull align 1 dereferenceable(1) ptr @"?_Get_first@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEBAAEBV?$allocator@D@2@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret ptr %3
}

; Function Attrs: noreturn
declare dso_local void @"?_Xlength_error@std@@YAXPEBD@Z"(ptr noundef) #17

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef nonnull align 1 dereferenceable(1) ptr @"?_Get_first@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAAAEAV?$allocator@D@2@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret ptr %3
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef i64 @"?_Calculate_growth@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@CA_K_K00@Z"(i64 noundef %0, i64 noundef %1, i64 noundef %2) #0 comdat align 2 {
  %4 = alloca i64, align 8
  %5 = alloca i64, align 8
  %6 = alloca i64, align 8
  %7 = alloca i64, align 8
  %8 = alloca i64, align 8
  %9 = alloca i64, align 8
  store i64 %2, ptr %5, align 8
  store i64 %1, ptr %6, align 8
  store i64 %0, ptr %7, align 8
  %10 = load i64, ptr %7, align 8
  %11 = or i64 %10, 15
  store i64 %11, ptr %8, align 8
  %12 = load i64, ptr %8, align 8
  %13 = load i64, ptr %5, align 8
  %14 = icmp ugt i64 %12, %13
  br i1 %14, label %15, label %17

15:                                               ; preds = %3
  %16 = load i64, ptr %5, align 8
  store i64 %16, ptr %4, align 8
  br label %33

17:                                               ; preds = %3
  %18 = load i64, ptr %6, align 8
  %19 = load i64, ptr %5, align 8
  %20 = load i64, ptr %6, align 8
  %21 = udiv i64 %20, 2
  %22 = sub i64 %19, %21
  %23 = icmp ugt i64 %18, %22
  br i1 %23, label %24, label %26

24:                                               ; preds = %17
  %25 = load i64, ptr %5, align 8
  store i64 %25, ptr %4, align 8
  br label %33

26:                                               ; preds = %17
  %27 = load i64, ptr %6, align 8
  %28 = load i64, ptr %6, align 8
  %29 = udiv i64 %28, 2
  %30 = add i64 %27, %29
  store i64 %30, ptr %9, align 8
  %31 = call noundef nonnull align 8 dereferenceable(8) ptr @"??$max@_K@std@@YAAEB_KAEB_K0@Z"(ptr noundef nonnull align 8 dereferenceable(8) %8, ptr noundef nonnull align 8 dereferenceable(8) %9) #3
  %32 = load i64, ptr %31, align 8
  store i64 %32, ptr %4, align 8
  br label %33

33:                                               ; preds = %26, %24, %15
  %34 = load i64, ptr %4, align 8
  ret i64 %34
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??$_Allocate_at_least_helper@V?$allocator@D@std@@@std@@YAPEADAEAV?$allocator@D@0@AEA_K@Z"(ptr noundef nonnull align 1 dereferenceable(1) %0, ptr noundef nonnull align 8 dereferenceable(8) %1) #10 comdat {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8, !nonnull !16
  %6 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %7 = load i64, ptr %6, align 8
  %8 = call noundef ptr @"?allocate@?$allocator@D@std@@QEAAPEAD_K@Z"(ptr noundef nonnull align 1 dereferenceable(1) %5, i64 noundef %7)
  ret ptr %8
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"?allocate@?$allocator@D@std@@QEAAPEAD_K@Z"(ptr noundef nonnull align 1 dereferenceable(1) %0, i64 noundef %1) #10 comdat align 2 {
  %3 = alloca i64, align 8
  %4 = alloca ptr, align 8
  store i64 %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = load i64, ptr %3, align 8
  %7 = call noundef i64 @"??$_Get_size_of_n@$00@std@@YA_K_K@Z"(i64 noundef %6)
  %8 = call noundef ptr @"??$_Allocate@$0BA@U_Default_allocate_traits@std@@$0A@@std@@YAPEAX_K@Z"(i64 noundef %7)
  ret ptr %8
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??$_Allocate@$0BA@U_Default_allocate_traits@std@@$0A@@std@@YAPEAX_K@Z"(i64 noundef %0) #10 comdat {
  %2 = alloca ptr, align 8
  %3 = alloca i64, align 8
  store i64 %0, ptr %3, align 8
  %4 = load i64, ptr %3, align 8
  %5 = icmp uge i64 %4, 4096
  br i1 %5, label %6, label %9

6:                                                ; preds = %1
  %7 = load i64, ptr %3, align 8
  %8 = call noundef ptr @"??$_Allocate_manually_vector_aligned@U_Default_allocate_traits@std@@@std@@YAPEAX_K@Z"(i64 noundef %7)
  store ptr %8, ptr %2, align 8
  br label %16

9:                                                ; preds = %1
  %10 = load i64, ptr %3, align 8
  %11 = icmp ne i64 %10, 0
  br i1 %11, label %12, label %15

12:                                               ; preds = %9
  %13 = load i64, ptr %3, align 8
  %14 = call noundef ptr @"?_Allocate@_Default_allocate_traits@std@@SAPEAX_K@Z"(i64 noundef %13)
  store ptr %14, ptr %2, align 8
  br label %16

15:                                               ; preds = %9
  store ptr null, ptr %2, align 8
  br label %16

16:                                               ; preds = %15, %12, %6
  %17 = load ptr, ptr %2, align 8
  ret ptr %17
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef i64 @"??$_Get_size_of_n@$00@std@@YA_K_K@Z"(i64 noundef %0) #0 comdat {
  %2 = alloca i64, align 8
  %3 = alloca i8, align 1
  store i64 %0, ptr %2, align 8
  store i8 0, ptr %3, align 1
  %4 = load i64, ptr %2, align 8
  %5 = mul i64 %4, 1
  ret i64 %5
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??$_Allocate_manually_vector_aligned@U_Default_allocate_traits@std@@@std@@YAPEAX_K@Z"(i64 noundef %0) #10 comdat {
  %2 = alloca i64, align 8
  %3 = alloca i64, align 8
  %4 = alloca i64, align 8
  %5 = alloca ptr, align 8
  store i64 %0, ptr %2, align 8
  %6 = load i64, ptr %2, align 8
  %7 = add i64 39, %6
  store i64 %7, ptr %3, align 8
  %8 = load i64, ptr %3, align 8
  %9 = load i64, ptr %2, align 8
  %10 = icmp ule i64 %8, %9
  br i1 %10, label %11, label %12

11:                                               ; preds = %1
  call void @"?_Throw_bad_array_new_length@std@@YAXXZ"() #22
  unreachable

12:                                               ; preds = %1
  %13 = load i64, ptr %3, align 8
  %14 = call noundef ptr @"?_Allocate@_Default_allocate_traits@std@@SAPEAX_K@Z"(i64 noundef %13)
  %15 = ptrtoint ptr %14 to i64
  store i64 %15, ptr %4, align 8
  br label %16

16:                                               ; preds = %12
  %17 = load i64, ptr %4, align 8
  %18 = icmp ne i64 %17, 0
  br i1 %18, label %19, label %20

19:                                               ; preds = %16
  br label %23

20:                                               ; preds = %16
  br label %21

21:                                               ; preds = %20
  call void @_invalid_parameter_noinfo_noreturn() #22
  unreachable

22:                                               ; No predecessors!
  br label %23

23:                                               ; preds = %22, %19
  br label %24

24:                                               ; preds = %23
  %25 = load i64, ptr %4, align 8
  %26 = add i64 %25, 39
  %27 = and i64 %26, -32
  %28 = inttoptr i64 %27 to ptr
  store ptr %28, ptr %5, align 8
  %29 = load i64, ptr %4, align 8
  %30 = load ptr, ptr %5, align 8
  %31 = getelementptr inbounds i64, ptr %30, i64 -1
  store i64 %29, ptr %31, align 8
  %32 = load ptr, ptr %5, align 8
  ret ptr %32
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"?_Allocate@_Default_allocate_traits@std@@SAPEAX_K@Z"(i64 noundef %0) #10 comdat align 2 {
  %2 = alloca i64, align 8
  store i64 %0, ptr %2, align 8
  %3 = load i64, ptr %2, align 8
  %4 = call noalias noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef %3) #21
  ret ptr %4
}

; Function Attrs: mustprogress noinline noreturn optnone sspstrong uwtable
define linkonce_odr dso_local void @"?_Throw_bad_array_new_length@std@@YAXXZ"() #20 comdat {
  %1 = alloca %"class.std::bad_array_new_length", align 8
  %2 = call noundef ptr @"??0bad_array_new_length@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %1) #3
  call void @_CxxThrowException(ptr %1, ptr @"_TI3?AVbad_array_new_length@std@@") #22
  unreachable
}

; Function Attrs: noreturn
declare dso_local void @_invalid_parameter_noinfo_noreturn() #17

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0bad_array_new_length@std@@QEAA@XZ"(ptr noundef nonnull returned align 8 dereferenceable(24) %0) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call noundef ptr @"??0bad_alloc@std@@AEAA@QEBD@Z"(ptr noundef nonnull align 8 dereferenceable(24) %3, ptr noundef @"??_C@_0BF@KINCDENJ@bad?5array?5new?5length?$AA@") #3
  invoke void @llvm.seh.scope.begin()
          to label %5 unwind label %7

5:                                                ; preds = %1
  store ptr @"??_7bad_array_new_length@std@@6B@", ptr %3, align 8
  invoke void @llvm.seh.scope.end()
          to label %6 unwind label %7

6:                                                ; preds = %5
  ret ptr %3

7:                                                ; preds = %5, %1
  %8 = cleanuppad within none []
  call void @"??1bad_alloc@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %3) #3 [ "funclet"(token %8) ]
  cleanupret from %8 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0bad_array_new_length@std@@QEAA@AEBV01@@Z"(ptr noundef nonnull returned align 8 dereferenceable(24) %0, ptr noundef nonnull align 8 dereferenceable(24) %1) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %7 = call noundef ptr @"??0bad_alloc@std@@QEAA@AEBV01@@Z"(ptr noundef nonnull align 8 dereferenceable(24) %5, ptr noundef nonnull align 8 dereferenceable(24) %6) #3
  invoke void @llvm.seh.scope.begin()
          to label %8 unwind label %10

8:                                                ; preds = %2
  store ptr @"??_7bad_array_new_length@std@@6B@", ptr %5, align 8
  invoke void @llvm.seh.scope.end()
          to label %9 unwind label %10

9:                                                ; preds = %8
  ret ptr %5

10:                                               ; preds = %8, %2
  %11 = cleanuppad within none []
  call void @"??1bad_alloc@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %5) #3 [ "funclet"(token %11) ]
  cleanupret from %11 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0bad_alloc@std@@QEAA@AEBV01@@Z"(ptr noundef nonnull returned align 8 dereferenceable(24) %0, ptr noundef nonnull align 8 dereferenceable(24) %1) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %7 = call noundef ptr @"??0exception@std@@QEAA@AEBV01@@Z"(ptr noundef nonnull align 8 dereferenceable(24) %5, ptr noundef nonnull align 8 dereferenceable(24) %6) #3
  invoke void @llvm.seh.scope.begin()
          to label %8 unwind label %10

8:                                                ; preds = %2
  store ptr @"??_7bad_alloc@std@@6B@", ptr %5, align 8
  invoke void @llvm.seh.scope.end()
          to label %9 unwind label %10

9:                                                ; preds = %8
  ret ptr %5

10:                                               ; preds = %8, %2
  %11 = cleanuppad within none []
  call void @"??1exception@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %5) #3 [ "funclet"(token %11) ]
  cleanupret from %11 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"??1bad_array_new_length@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  invoke void @llvm.seh.scope.begin()
          to label %4 unwind label %6

4:                                                ; preds = %1
  invoke void @llvm.seh.scope.end()
          to label %5 unwind label %6

5:                                                ; preds = %4
  call void @"??1bad_alloc@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %3) #3
  ret void

6:                                                ; preds = %4, %1
  %7 = cleanuppad within none []
  call void @"??1bad_alloc@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %3) #3 [ "funclet"(token %7) ]
  cleanupret from %7 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0bad_alloc@std@@AEAA@QEBD@Z"(ptr noundef nonnull returned align 8 dereferenceable(24) %0, ptr noundef %1) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = load ptr, ptr %3, align 8
  %7 = call noundef ptr @"??0exception@std@@QEAA@QEBDH@Z"(ptr noundef nonnull align 8 dereferenceable(24) %5, ptr noundef %6, i32 noundef 1) #3
  invoke void @llvm.seh.scope.begin()
          to label %8 unwind label %10

8:                                                ; preds = %2
  store ptr @"??_7bad_alloc@std@@6B@", ptr %5, align 8
  invoke void @llvm.seh.scope.end()
          to label %9 unwind label %10

9:                                                ; preds = %8
  ret ptr %5

10:                                               ; preds = %8, %2
  %11 = cleanuppad within none []
  call void @"??1exception@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %5) #3 [ "funclet"(token %11) ]
  cleanupret from %11 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"??1bad_alloc@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  invoke void @llvm.seh.scope.begin()
          to label %4 unwind label %6

4:                                                ; preds = %1
  invoke void @llvm.seh.scope.end()
          to label %5 unwind label %6

5:                                                ; preds = %4
  call void @"??1exception@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %3) #3
  ret void

6:                                                ; preds = %4, %1
  %7 = cleanuppad within none []
  call void @"??1exception@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %3) #3 [ "funclet"(token %7) ]
  cleanupret from %7 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??_Gbad_array_new_length@std@@UEAAPEAXI@Z"(ptr noundef nonnull align 8 dereferenceable(24) %0, i32 noundef %1) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  %5 = alloca ptr, align 8
  store i32 %1, ptr %4, align 4
  store ptr %0, ptr %5, align 8
  %6 = load ptr, ptr %5, align 8
  store ptr %6, ptr %3, align 8
  %7 = load i32, ptr %4, align 4
  invoke void @llvm.seh.scope.begin()
          to label %8 unwind label %14

8:                                                ; preds = %2
  call void @"??1bad_array_new_length@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %6) #3
  invoke void @llvm.seh.scope.end()
          to label %9 unwind label %14

9:                                                ; preds = %8
  %10 = icmp eq i32 %7, 0
  br i1 %10, label %12, label %11

11:                                               ; preds = %9
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %6, i64 noundef 24) #23
  br label %12

12:                                               ; preds = %11, %9
  %13 = load ptr, ptr %3, align 8
  ret ptr %13

14:                                               ; preds = %8, %2
  %15 = cleanuppad within none []
  %16 = icmp eq i32 %7, 0
  br i1 %16, label %18, label %17

17:                                               ; preds = %14
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %6, i64 noundef 24) #23 [ "funclet"(token %15) ]
  br label %18

18:                                               ; preds = %17, %14
  cleanupret from %15 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0exception@std@@QEAA@QEBDH@Z"(ptr noundef nonnull returned align 8 dereferenceable(24) %0, ptr noundef %1, i32 noundef %2) unnamed_addr #0 comdat align 2 {
  %4 = alloca i32, align 4
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  store i32 %2, ptr %4, align 4
  store ptr %1, ptr %5, align 8
  store ptr %0, ptr %6, align 8
  %7 = load ptr, ptr %6, align 8
  store ptr @"??_7exception@std@@6B@", ptr %7, align 8
  %8 = getelementptr inbounds nuw %"class.std::exception", ptr %7, i32 0, i32 1
  call void @llvm.memset.p0.i64(ptr align 8 %8, i8 0, i64 16, i1 false)
  %9 = load ptr, ptr %5, align 8
  %10 = getelementptr inbounds nuw %"class.std::exception", ptr %7, i32 0, i32 1
  %11 = getelementptr inbounds nuw %struct.__std_exception_data, ptr %10, i32 0, i32 0
  store ptr %9, ptr %11, align 8
  ret ptr %7
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??_Gbad_alloc@std@@UEAAPEAXI@Z"(ptr noundef nonnull align 8 dereferenceable(24) %0, i32 noundef %1) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  %5 = alloca ptr, align 8
  store i32 %1, ptr %4, align 4
  store ptr %0, ptr %5, align 8
  %6 = load ptr, ptr %5, align 8
  store ptr %6, ptr %3, align 8
  %7 = load i32, ptr %4, align 4
  invoke void @llvm.seh.scope.begin()
          to label %8 unwind label %14

8:                                                ; preds = %2
  call void @"??1bad_alloc@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %6) #3
  invoke void @llvm.seh.scope.end()
          to label %9 unwind label %14

9:                                                ; preds = %8
  %10 = icmp eq i32 %7, 0
  br i1 %10, label %12, label %11

11:                                               ; preds = %9
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %6, i64 noundef 24) #23
  br label %12

12:                                               ; preds = %11, %9
  %13 = load ptr, ptr %3, align 8
  ret ptr %13

14:                                               ; preds = %8, %2
  %15 = cleanuppad within none []
  %16 = icmp eq i32 %7, 0
  br i1 %16, label %18, label %17

17:                                               ; preds = %14
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %6, i64 noundef 24) #23 [ "funclet"(token %15) ]
  br label %18

18:                                               ; preds = %17, %14
  cleanupret from %15 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef i64 @"??$_Convert_size@_K_K@std@@YA_K_K@Z"(i64 noundef %0) #0 comdat {
  %2 = alloca i64, align 8
  store i64 %0, ptr %2, align 8
  %3 = load i64, ptr %2, align 8
  ret i64 %3
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0error_condition@std@@QEAA@HAEBVerror_category@1@@Z"(ptr noundef nonnull returned align 8 dereferenceable(16) %0, i32 noundef %1, ptr noundef nonnull align 8 dereferenceable(16) %2) unnamed_addr #0 comdat align 2 {
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  %6 = alloca ptr, align 8
  store ptr %2, ptr %4, align 8
  store i32 %1, ptr %5, align 4
  store ptr %0, ptr %6, align 8
  %7 = load ptr, ptr %6, align 8
  %8 = getelementptr inbounds nuw %"class.std::error_condition", ptr %7, i32 0, i32 0
  %9 = load i32, ptr %5, align 4
  store i32 %9, ptr %8, align 8
  %10 = getelementptr inbounds nuw %"class.std::error_condition", ptr %7, i32 0, i32 1
  %11 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  store ptr %11, ptr %10, align 8
  ret ptr %7
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef zeroext i1 @"??8error_category@std@@QEBA_NAEBV01@@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1) #0 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr inbounds nuw %"class.std::error_category", ptr %5, i32 0, i32 1
  %7 = call noundef i64 @"??$_Bit_cast@_KT_Addr_storage@error_category@std@@$0A@@std@@YA_KAEBT_Addr_storage@error_category@0@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %6) #3
  %8 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %9 = getelementptr inbounds nuw %"class.std::error_category", ptr %8, i32 0, i32 1
  %10 = call noundef i64 @"??$_Bit_cast@_KT_Addr_storage@error_category@std@@$0A@@std@@YA_KAEBT_Addr_storage@error_category@0@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %9) #3
  %11 = icmp eq i64 %7, %10
  ret i1 %11
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(16) ptr @"?category@error_code@std@@QEBAAEBVerror_category@2@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"class.std::error_code", ptr %3, i32 0, i32 1
  %5 = load ptr, ptr %4, align 8
  ret ptr %5
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef i32 @"?value@error_code@std@@QEBAHXZ"(ptr noundef nonnull align 8 dereferenceable(16) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"class.std::error_code", ptr %3, i32 0, i32 0
  %5 = load i32, ptr %4, align 8
  ret i32 %5
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef i64 @"??$_Bit_cast@_KT_Addr_storage@error_category@std@@$0A@@std@@YA_KAEBT_Addr_storage@error_category@0@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0) #0 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8, !nonnull !16, !align !17
  %4 = load i64, ptr %3, align 8
  ret i64 %4
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef zeroext i1 @"??8std@@YA_NAEBVerror_condition@0@0@Z"(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1) #0 comdat {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %6 = call noundef nonnull align 8 dereferenceable(16) ptr @"?category@error_condition@std@@QEBAAEBVerror_category@2@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %5) #3
  %7 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %8 = call noundef nonnull align 8 dereferenceable(16) ptr @"?category@error_condition@std@@QEBAAEBVerror_category@2@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %7) #3
  %9 = call noundef zeroext i1 @"??8error_category@std@@QEBA_NAEBV01@@Z"(ptr noundef nonnull align 8 dereferenceable(16) %6, ptr noundef nonnull align 8 dereferenceable(16) %8) #3
  br i1 %9, label %10, label %16

10:                                               ; preds = %2
  %11 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %12 = call noundef i32 @"?value@error_condition@std@@QEBAHXZ"(ptr noundef nonnull align 8 dereferenceable(16) %11) #3
  %13 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %14 = call noundef i32 @"?value@error_condition@std@@QEBAHXZ"(ptr noundef nonnull align 8 dereferenceable(16) %13) #3
  %15 = icmp eq i32 %12, %14
  br label %16

16:                                               ; preds = %10, %2
  %17 = phi i1 [ false, %2 ], [ %15, %10 ]
  ret i1 %17
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(16) ptr @"?category@error_condition@std@@QEBAAEBVerror_category@2@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"class.std::error_condition", ptr %3, i32 0, i32 1
  %5 = load ptr, ptr %4, align 8
  ret ptr %5
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef i32 @"?value@error_condition@std@@QEBAHXZ"(ptr noundef nonnull align 8 dereferenceable(16) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"class.std::error_condition", ptr %3, i32 0, i32 0
  %5 = load i32, ptr %4, align 8
  ret i32 %5
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0system_error@std@@QEAA@Verror_code@1@PEBD@Z"(ptr noundef nonnull returned align 8 dereferenceable(40) %0, ptr noundef %1, ptr noundef %2) unnamed_addr #10 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca %"class.std::basic_string", align 8
  %8 = alloca %"class.std::error_code", align 8
  store ptr %2, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store ptr %0, ptr %6, align 8
  %9 = load ptr, ptr %6, align 8
  %10 = load ptr, ptr %4, align 8
  %11 = call noundef ptr @"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD@Z"(ptr noundef nonnull align 8 dereferenceable(32) %7, ptr noundef %10)
  invoke void @llvm.seh.scope.begin()
          to label %12 unwind label %18

12:                                               ; preds = %3
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %8, ptr align 8 %1, i64 16, i1 false)
  %13 = invoke noundef ptr @"??0_System_error@std@@IEAA@Verror_code@1@AEBV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@1@@Z"(ptr noundef nonnull align 8 dereferenceable(40) %9, ptr noundef %8, ptr noundef nonnull align 8 dereferenceable(32) %7)
          to label %14 unwind label %18

14:                                               ; preds = %12
  invoke void @llvm.seh.scope.end()
          to label %15 unwind label %18

15:                                               ; preds = %14
  call void @"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %7) #3
  invoke void @llvm.seh.scope.begin()
          to label %16 unwind label %20

16:                                               ; preds = %15
  store ptr @"??_7system_error@std@@6B@", ptr %9, align 8
  invoke void @llvm.seh.scope.end()
          to label %17 unwind label %20

17:                                               ; preds = %16
  ret ptr %9

18:                                               ; preds = %14, %12, %3
  %19 = cleanuppad within none []
  call void @"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %7) #3 [ "funclet"(token %19) ]
  cleanupret from %19 unwind to caller

20:                                               ; preds = %16, %15
  %21 = cleanuppad within none []
  call void @"??1_System_error@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(40) %9) #3 [ "funclet"(token %21) ]
  cleanupret from %21 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"??1system_error@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(40) %0) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  invoke void @llvm.seh.scope.begin()
          to label %4 unwind label %6

4:                                                ; preds = %1
  invoke void @llvm.seh.scope.end()
          to label %5 unwind label %6

5:                                                ; preds = %4
  call void @"??1_System_error@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(40) %3) #3
  ret void

6:                                                ; preds = %4, %1
  %7 = cleanuppad within none []
  call void @"??1_System_error@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(40) %3) #3 [ "funclet"(token %7) ]
  cleanupret from %7 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??_Gfailure@ios_base@std@@UEAAPEAXI@Z"(ptr noundef nonnull align 8 dereferenceable(40) %0, i32 noundef %1) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  %5 = alloca ptr, align 8
  store i32 %1, ptr %4, align 4
  store ptr %0, ptr %5, align 8
  %6 = load ptr, ptr %5, align 8
  store ptr %6, ptr %3, align 8
  %7 = load i32, ptr %4, align 4
  invoke void @llvm.seh.scope.begin()
          to label %8 unwind label %14

8:                                                ; preds = %2
  call void @"??1failure@ios_base@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(40) %6) #3
  invoke void @llvm.seh.scope.end()
          to label %9 unwind label %14

9:                                                ; preds = %8
  %10 = icmp eq i32 %7, 0
  br i1 %10, label %12, label %11

11:                                               ; preds = %9
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %6, i64 noundef 40) #23
  br label %12

12:                                               ; preds = %11, %9
  %13 = load ptr, ptr %3, align 8
  ret ptr %13

14:                                               ; preds = %8, %2
  %15 = cleanuppad within none []
  %16 = icmp eq i32 %7, 0
  br i1 %16, label %18, label %17

17:                                               ; preds = %14
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %6, i64 noundef 40) #23 [ "funclet"(token %15) ]
  br label %18

18:                                               ; preds = %17, %14
  cleanupret from %15 unwind to caller
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0_System_error@std@@IEAA@Verror_code@1@AEBV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@1@@Z"(ptr noundef nonnull returned align 8 dereferenceable(40) %0, ptr noundef %1, ptr noundef nonnull align 8 dereferenceable(32) %2) unnamed_addr #10 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca %"class.std::basic_string", align 8
  %8 = alloca %"class.std::basic_string", align 8
  %9 = alloca %"class.std::error_code", align 8
  store ptr %2, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store ptr %0, ptr %6, align 8
  %10 = load ptr, ptr %6, align 8
  %11 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %12 = call noundef ptr @"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@AEBV01@@Z"(ptr noundef nonnull align 8 dereferenceable(32) %8, ptr noundef nonnull align 8 dereferenceable(32) %11)
  invoke void @llvm.seh.scope.begin()
          to label %13 unwind label %22

13:                                               ; preds = %3
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %9, ptr align 8 %1, i64 16, i1 false)
  invoke void @llvm.seh.scope.end()
          to label %14 unwind label %22

14:                                               ; preds = %13
  call void @"?_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z"(ptr dead_on_unwind writable sret(%"class.std::basic_string") align 8 %7, ptr noundef %9, ptr noundef %8)
  invoke void @llvm.seh.scope.begin()
          to label %15 unwind label %24

15:                                               ; preds = %14
  %16 = invoke noundef ptr @"??0runtime_error@std@@QEAA@AEBV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@1@@Z"(ptr noundef nonnull align 8 dereferenceable(24) %10, ptr noundef nonnull align 8 dereferenceable(32) %7)
          to label %17 unwind label %24

17:                                               ; preds = %15
  invoke void @llvm.seh.scope.end()
          to label %18 unwind label %24

18:                                               ; preds = %17
  call void @"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %7) #3
  invoke void @llvm.seh.scope.begin()
          to label %19 unwind label %26

19:                                               ; preds = %18
  store ptr @"??_7_System_error@std@@6B@", ptr %10, align 8
  %20 = getelementptr inbounds nuw %"class.std::_System_error", ptr %10, i32 0, i32 1
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %20, ptr align 8 %1, i64 16, i1 false)
  invoke void @llvm.seh.scope.end()
          to label %21 unwind label %26

21:                                               ; preds = %19
  ret ptr %10

22:                                               ; preds = %13, %3
  %23 = cleanuppad within none []
  call void @"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %8) #3 [ "funclet"(token %23) ]
  cleanupret from %23 unwind to caller

24:                                               ; preds = %17, %15, %14
  %25 = cleanuppad within none []
  call void @"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %7) #3 [ "funclet"(token %25) ]
  cleanupret from %25 unwind to caller

26:                                               ; preds = %19, %18
  %27 = cleanuppad within none []
  call void @"??1runtime_error@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %10) #3 [ "funclet"(token %27) ]
  cleanupret from %27 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %0) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  invoke void @llvm.seh.scope.begin()
          to label %4 unwind label %7

4:                                                ; preds = %1
  call void @"?_Tidy_deallocate@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXXZ"(ptr noundef nonnull align 8 dereferenceable(32) %3) #3
  invoke void @llvm.seh.scope.end()
          to label %5 unwind label %7

5:                                                ; preds = %4
  %6 = getelementptr inbounds nuw %"class.std::basic_string", ptr %3, i32 0, i32 0
  call void @"??1?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %6) #3
  ret void

7:                                                ; preds = %4, %1
  %8 = cleanuppad within none []
  %9 = getelementptr inbounds nuw %"class.std::basic_string", ptr %3, i32 0, i32 0
  call void @"??1?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %9) #3 [ "funclet"(token %8) ]
  cleanupret from %8 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"??1_System_error@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(40) %0) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  invoke void @llvm.seh.scope.begin()
          to label %4 unwind label %6

4:                                                ; preds = %1
  invoke void @llvm.seh.scope.end()
          to label %5 unwind label %6

5:                                                ; preds = %4
  call void @"??1runtime_error@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %3) #3
  ret void

6:                                                ; preds = %4, %1
  %7 = cleanuppad within none []
  call void @"??1runtime_error@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %3) #3 [ "funclet"(token %7) ]
  cleanupret from %7 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??_Gsystem_error@std@@UEAAPEAXI@Z"(ptr noundef nonnull align 8 dereferenceable(40) %0, i32 noundef %1) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  %5 = alloca ptr, align 8
  store i32 %1, ptr %4, align 4
  store ptr %0, ptr %5, align 8
  %6 = load ptr, ptr %5, align 8
  store ptr %6, ptr %3, align 8
  %7 = load i32, ptr %4, align 4
  invoke void @llvm.seh.scope.begin()
          to label %8 unwind label %14

8:                                                ; preds = %2
  call void @"??1system_error@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(40) %6) #3
  invoke void @llvm.seh.scope.end()
          to label %9 unwind label %14

9:                                                ; preds = %8
  %10 = icmp eq i32 %7, 0
  br i1 %10, label %12, label %11

11:                                               ; preds = %9
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %6, i64 noundef 40) #23
  br label %12

12:                                               ; preds = %11, %9
  %13 = load ptr, ptr %3, align 8
  ret ptr %13

14:                                               ; preds = %8, %2
  %15 = cleanuppad within none []
  %16 = icmp eq i32 %7, 0
  br i1 %16, label %18, label %17

17:                                               ; preds = %14
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %6, i64 noundef 40) #23 [ "funclet"(token %15) ]
  br label %18

18:                                               ; preds = %17, %14
  cleanupret from %15 unwind to caller
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local void @"?_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z"(ptr dead_on_unwind noalias writable sret(%"class.std::basic_string") align 8 %0, ptr noundef %1, ptr noundef %2) #10 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca %"class.std::basic_string", align 8
  store ptr %0, ptr %4, align 8
  store ptr %2, ptr %5, align 8
  invoke void @llvm.seh.scope.begin()
          to label %8 unwind label %24

8:                                                ; preds = %3
  store ptr %1, ptr %6, align 8
  %9 = call noundef zeroext i1 @"?empty@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_NXZ"(ptr noundef nonnull align 8 dereferenceable(32) %2) #3
  br i1 %9, label %13, label %10

10:                                               ; preds = %8
  %11 = invoke noundef nonnull align 8 dereferenceable(32) ptr @"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@QEBD@Z"(ptr noundef nonnull align 8 dereferenceable(32) %2, ptr noundef @"??_C@_02LMMGGCAJ@?3?5?$AA@")
          to label %12 unwind label %24

12:                                               ; preds = %10
  br label %13

13:                                               ; preds = %12, %8
  invoke void @"?message@error_code@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %1, ptr dead_on_unwind writable sret(%"class.std::basic_string") align 8 %7)
          to label %14 unwind label %24

14:                                               ; preds = %13
  invoke void @llvm.seh.scope.begin()
          to label %15 unwind label %20

15:                                               ; preds = %14
  %16 = invoke noundef nonnull align 8 dereferenceable(32) ptr @"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@AEBV12@@Z"(ptr noundef nonnull align 8 dereferenceable(32) %2, ptr noundef nonnull align 8 dereferenceable(32) %7)
          to label %17 unwind label %20

17:                                               ; preds = %15
  invoke void @llvm.seh.scope.end()
          to label %18 unwind label %20

18:                                               ; preds = %17
  call void @"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %7) #3
  %19 = call noundef ptr @"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@$$QEAV01@@Z"(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(32) %2) #3
  invoke void @llvm.seh.scope.end()
          to label %23 unwind label %24

20:                                               ; preds = %17, %15, %14
  %21 = cleanuppad within none []
  invoke void @llvm.seh.scope.end() [ "funclet"(token %21) ]
          to label %22 unwind label %24

22:                                               ; preds = %20
  call void @"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %7) #3 [ "funclet"(token %21) ]
  cleanupret from %21 unwind label %24

23:                                               ; preds = %18
  call void @"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %2) #3
  ret void

24:                                               ; preds = %18, %22, %20, %13, %10, %3
  %25 = cleanuppad within none []
  call void @"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %2) #3 [ "funclet"(token %25) ]
  cleanupret from %25 unwind to caller
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@AEBV01@@Z"(ptr noundef nonnull returned align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(32) %1) unnamed_addr #10 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca %"class.std::allocator", align 1
  %6 = alloca %"struct.std::_One_then_variadic_args_t", align 1
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %7 = load ptr, ptr %4, align 8
  %8 = getelementptr inbounds nuw %"class.std::basic_string", ptr %7, i32 0, i32 0
  %9 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %10 = call noundef nonnull align 1 dereferenceable(1) ptr @"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBAAEBV?$allocator@D@2@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %9) #3
  call void @"?select_on_container_copy_construction@?$_Default_allocator_traits@V?$allocator@D@std@@@std@@SA?AV?$allocator@D@2@AEBV32@@Z"(ptr dead_on_unwind writable sret(%"class.std::allocator") align 1 %5, ptr noundef nonnull align 1 dereferenceable(1) %10)
  %11 = getelementptr inbounds nuw %"struct.std::_One_then_variadic_args_t", ptr %6, i32 0, i32 0
  %12 = load i8, ptr %11, align 1
  %13 = call noundef ptr @"??$?0V?$allocator@D@std@@$$V@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@U_One_then_variadic_args_t@1@$$QEAV?$allocator@D@1@@Z"(ptr noundef nonnull align 8 dereferenceable(32) %8, i8 %12, ptr noundef nonnull align 1 dereferenceable(1) %5) #3
  invoke void @llvm.seh.scope.begin()
          to label %14 unwind label %26

14:                                               ; preds = %2
  %15 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %16 = getelementptr inbounds nuw %"class.std::basic_string", ptr %15, i32 0, i32 0
  %17 = getelementptr inbounds nuw %"class.std::_Compressed_pair.10", ptr %16, i32 0, i32 0
  %18 = getelementptr inbounds nuw %"class.std::_String_val", ptr %17, i32 0, i32 1
  %19 = load i64, ptr %18, align 8
  %20 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %21 = getelementptr inbounds nuw %"class.std::basic_string", ptr %20, i32 0, i32 0
  %22 = getelementptr inbounds nuw %"class.std::_Compressed_pair.10", ptr %21, i32 0, i32 0
  %23 = call noundef ptr @"?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBAPEBDXZ"(ptr noundef nonnull align 8 dereferenceable(32) %22) #3
  invoke void @"??$_Construct@$01PEBD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXQEBD_K@Z"(ptr noundef nonnull align 8 dereferenceable(32) %7, ptr noundef %23, i64 noundef %19)
          to label %24 unwind label %26

24:                                               ; preds = %14
  invoke void @llvm.seh.scope.end()
          to label %25 unwind label %26

25:                                               ; preds = %24
  ret ptr %7

26:                                               ; preds = %24, %14, %2
  %27 = cleanuppad within none []
  call void @"??1?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %8) #3 [ "funclet"(token %27) ]
  cleanupret from %27 unwind to caller
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0runtime_error@std@@QEAA@AEBV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@1@@Z"(ptr noundef nonnull returned align 8 dereferenceable(24) %0, ptr noundef nonnull align 8 dereferenceable(32) %1) unnamed_addr #10 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %7 = call noundef ptr @"?c_str@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBAPEBDXZ"(ptr noundef nonnull align 8 dereferenceable(32) %6) #3
  %8 = call noundef ptr @"??0exception@std@@QEAA@QEBD@Z"(ptr noundef nonnull align 8 dereferenceable(24) %5, ptr noundef %7) #3
  invoke void @llvm.seh.scope.begin()
          to label %9 unwind label %11

9:                                                ; preds = %2
  store ptr @"??_7runtime_error@std@@6B@", ptr %5, align 8
  invoke void @llvm.seh.scope.end()
          to label %10 unwind label %11

10:                                               ; preds = %9
  ret ptr %5

11:                                               ; preds = %9, %2
  %12 = cleanuppad within none []
  call void @"??1exception@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %5) #3 [ "funclet"(token %12) ]
  cleanupret from %12 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??_G_System_error@std@@UEAAPEAXI@Z"(ptr noundef nonnull align 8 dereferenceable(40) %0, i32 noundef %1) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  %5 = alloca ptr, align 8
  store i32 %1, ptr %4, align 4
  store ptr %0, ptr %5, align 8
  %6 = load ptr, ptr %5, align 8
  store ptr %6, ptr %3, align 8
  %7 = load i32, ptr %4, align 4
  invoke void @llvm.seh.scope.begin()
          to label %8 unwind label %14

8:                                                ; preds = %2
  call void @"??1_System_error@std@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(40) %6) #3
  invoke void @llvm.seh.scope.end()
          to label %9 unwind label %14

9:                                                ; preds = %8
  %10 = icmp eq i32 %7, 0
  br i1 %10, label %12, label %11

11:                                               ; preds = %9
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %6, i64 noundef 40) #23
  br label %12

12:                                               ; preds = %11, %9
  %13 = load ptr, ptr %3, align 8
  ret ptr %13

14:                                               ; preds = %8, %2
  %15 = cleanuppad within none []
  %16 = icmp eq i32 %7, 0
  br i1 %16, label %18, label %17

17:                                               ; preds = %14
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %6, i64 noundef 40) #23 [ "funclet"(token %15) ]
  br label %18

18:                                               ; preds = %17, %14
  cleanupret from %15 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef zeroext i1 @"?empty@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_NXZ"(ptr noundef nonnull align 8 dereferenceable(32) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"class.std::basic_string", ptr %3, i32 0, i32 0
  %5 = getelementptr inbounds nuw %"class.std::_Compressed_pair.10", ptr %4, i32 0, i32 0
  %6 = getelementptr inbounds nuw %"class.std::_String_val", ptr %5, i32 0, i32 1
  %7 = load i64, ptr %6, align 8
  %8 = icmp eq i64 %7, 0
  ret i1 %8
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(32) ptr @"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@QEBD@Z"(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef %1) #10 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = load ptr, ptr %3, align 8
  %7 = call noundef i64 @"?length@?$_Narrow_char_traits@DH@std@@SA_KQEBD@Z"(ptr noundef %6) #3
  %8 = call noundef i64 @"??$_Convert_size@_K_K@std@@YA_K_K@Z"(i64 noundef %7) #3
  %9 = load ptr, ptr %3, align 8
  %10 = call noundef nonnull align 8 dereferenceable(32) ptr @"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@QEBD_K@Z"(ptr noundef nonnull align 8 dereferenceable(32) %5, ptr noundef %9, i64 noundef %8)
  ret ptr %10
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(32) ptr @"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@AEBV12@@Z"(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(32) %1) #10 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %7 = getelementptr inbounds nuw %"class.std::basic_string", ptr %6, i32 0, i32 0
  %8 = getelementptr inbounds nuw %"class.std::_Compressed_pair.10", ptr %7, i32 0, i32 0
  %9 = getelementptr inbounds nuw %"class.std::_String_val", ptr %8, i32 0, i32 1
  %10 = load i64, ptr %9, align 8
  %11 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %12 = getelementptr inbounds nuw %"class.std::basic_string", ptr %11, i32 0, i32 0
  %13 = getelementptr inbounds nuw %"class.std::_Compressed_pair.10", ptr %12, i32 0, i32 0
  %14 = call noundef ptr @"?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBAPEBDXZ"(ptr noundef nonnull align 8 dereferenceable(32) %13) #3
  %15 = call noundef nonnull align 8 dereferenceable(32) ptr @"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@QEBD_K@Z"(ptr noundef nonnull align 8 dereferenceable(32) %5, ptr noundef %14, i64 noundef %10)
  ret ptr %15
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local void @"?message@error_code@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr dead_on_unwind noalias writable sret(%"class.std::basic_string") align 8 %1) #10 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = call noundef nonnull align 8 dereferenceable(16) ptr @"?category@error_code@std@@QEBAAEBVerror_category@2@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %5) #3
  %7 = call noundef i32 @"?value@error_code@std@@QEBAHXZ"(ptr noundef nonnull align 8 dereferenceable(16) %5) #3
  %8 = load ptr, ptr %6, align 8
  %9 = getelementptr inbounds ptr, ptr %8, i64 2
  %10 = load ptr, ptr %9, align 8
  call void %10(ptr noundef nonnull align 8 dereferenceable(16) %6, ptr dead_on_unwind writable sret(%"class.std::basic_string") align 8 %1, i32 noundef %7)
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@$$QEAV01@@Z"(ptr noundef nonnull returned align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(32) %1) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca %"struct.std::_One_then_variadic_args_t", align 1
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = getelementptr inbounds nuw %"class.std::basic_string", ptr %6, i32 0, i32 0
  %8 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %9 = call noundef nonnull align 1 dereferenceable(1) ptr @"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV?$allocator@D@2@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %8) #3
  %10 = getelementptr inbounds nuw %"struct.std::_One_then_variadic_args_t", ptr %5, i32 0, i32 0
  %11 = load i8, ptr %10, align 1
  %12 = call noundef ptr @"??$?0V?$allocator@D@std@@$$V@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@U_One_then_variadic_args_t@1@$$QEAV?$allocator@D@1@@Z"(ptr noundef nonnull align 8 dereferenceable(32) %7, i8 %11, ptr noundef nonnull align 1 dereferenceable(1) %9) #3
  invoke void @llvm.seh.scope.begin()
          to label %13 unwind label %18

13:                                               ; preds = %2
  %14 = getelementptr inbounds nuw %"class.std::basic_string", ptr %6, i32 0, i32 0
  %15 = getelementptr inbounds nuw %"class.std::_Compressed_pair.10", ptr %14, i32 0, i32 0
  call void @"?_Alloc_proxy@_Container_base0@std@@QEAAXAEBU_Fake_allocator@2@@Z"(ptr noundef nonnull align 1 dereferenceable(1) %15, ptr noundef nonnull align 1 dereferenceable(1) @"?_Fake_alloc@std@@3U_Fake_allocator@1@B") #3
  %16 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  call void @"?_Take_contents@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXAEAV12@@Z"(ptr noundef nonnull align 8 dereferenceable(32) %6, ptr noundef nonnull align 8 dereferenceable(32) %16) #3
  invoke void @llvm.seh.scope.end()
          to label %17 unwind label %18

17:                                               ; preds = %13
  ret ptr %6

18:                                               ; preds = %13, %2
  %19 = cleanuppad within none []
  call void @"??1?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %7) #3 [ "funclet"(token %19) ]
  cleanupret from %19 unwind to caller
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(32) ptr @"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@QEBD_K@Z"(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef %1, i64 noundef %2) #10 comdat align 2 {
  %4 = alloca ptr, align 8
  %5 = alloca i64, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca i64, align 8
  %9 = alloca ptr, align 8
  %10 = alloca i8, align 1
  %11 = alloca %class.anon.11, align 1
  store i64 %2, ptr %5, align 8
  store ptr %1, ptr %6, align 8
  store ptr %0, ptr %7, align 8
  %12 = load ptr, ptr %7, align 8
  %13 = getelementptr inbounds nuw %"class.std::basic_string", ptr %12, i32 0, i32 0
  %14 = getelementptr inbounds nuw %"class.std::_Compressed_pair.10", ptr %13, i32 0, i32 0
  %15 = getelementptr inbounds nuw %"class.std::_String_val", ptr %14, i32 0, i32 1
  %16 = load i64, ptr %15, align 8
  store i64 %16, ptr %8, align 8
  %17 = load i64, ptr %5, align 8
  %18 = getelementptr inbounds nuw %"class.std::basic_string", ptr %12, i32 0, i32 0
  %19 = getelementptr inbounds nuw %"class.std::_Compressed_pair.10", ptr %18, i32 0, i32 0
  %20 = getelementptr inbounds nuw %"class.std::_String_val", ptr %19, i32 0, i32 2
  %21 = load i64, ptr %20, align 8
  %22 = load i64, ptr %8, align 8
  %23 = sub i64 %21, %22
  %24 = icmp ule i64 %17, %23
  br i1 %24, label %25, label %46

25:                                               ; preds = %3
  %26 = load i64, ptr %8, align 8
  %27 = load i64, ptr %5, align 8
  %28 = add i64 %26, %27
  %29 = getelementptr inbounds nuw %"class.std::basic_string", ptr %12, i32 0, i32 0
  %30 = getelementptr inbounds nuw %"class.std::_Compressed_pair.10", ptr %29, i32 0, i32 0
  %31 = getelementptr inbounds nuw %"class.std::_String_val", ptr %30, i32 0, i32 1
  store i64 %28, ptr %31, align 8
  %32 = getelementptr inbounds nuw %"class.std::basic_string", ptr %12, i32 0, i32 0
  %33 = getelementptr inbounds nuw %"class.std::_Compressed_pair.10", ptr %32, i32 0, i32 0
  %34 = call noundef ptr @"?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAAPEADXZ"(ptr noundef nonnull align 8 dereferenceable(32) %33) #3
  store ptr %34, ptr %9, align 8
  %35 = load i64, ptr %5, align 8
  %36 = load ptr, ptr %6, align 8
  %37 = load ptr, ptr %9, align 8
  %38 = load i64, ptr %8, align 8
  %39 = getelementptr inbounds nuw i8, ptr %37, i64 %38
  %40 = call noundef ptr @"?move@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z"(ptr noundef %39, ptr noundef %36, i64 noundef %35) #3
  store i8 0, ptr %10, align 1
  %41 = load ptr, ptr %9, align 8
  %42 = load i64, ptr %8, align 8
  %43 = load i64, ptr %5, align 8
  %44 = add i64 %42, %43
  %45 = getelementptr inbounds nuw i8, ptr %41, i64 %44
  call void @"?assign@?$_Narrow_char_traits@DH@std@@SAXAEADAEBD@Z"(ptr noundef nonnull align 1 dereferenceable(1) %45, ptr noundef nonnull align 1 dereferenceable(1) %10) #3
  store ptr %12, ptr %4, align 8
  br label %53

46:                                               ; preds = %3
  %47 = load i64, ptr %5, align 8
  %48 = load ptr, ptr %6, align 8
  %49 = load i64, ptr %5, align 8
  %50 = getelementptr inbounds nuw %class.anon.11, ptr %11, i32 0, i32 0
  %51 = load i8, ptr %50, align 1
  %52 = call noundef nonnull align 8 dereferenceable(32) ptr @"??$_Reallocate_grow_by@V<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@QEBD_K@Z@PEBD_K@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??append@01@QEAAAEAV01@QEBD0@Z@PEBD_K@Z"(ptr noundef nonnull align 8 dereferenceable(32) %12, i64 noundef %49, i8 %51, ptr noundef %48, i64 noundef %47)
  store ptr %52, ptr %4, align 8
  br label %53

53:                                               ; preds = %46, %25
  %54 = load ptr, ptr %4, align 8
  ret ptr %54
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAAPEADXZ"(ptr noundef nonnull align 8 dereferenceable(32) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr inbounds nuw %"class.std::_String_val", ptr %4, i32 0, i32 0
  %6 = getelementptr inbounds [16 x i8], ptr %5, i64 0, i64 0
  store ptr %6, ptr %3, align 8
  %7 = call noundef zeroext i1 @"?_Large_mode_engaged@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBA_NXZ"(ptr noundef nonnull align 8 dereferenceable(32) %4) #3
  br i1 %7, label %8, label %12

8:                                                ; preds = %1
  %9 = getelementptr inbounds nuw %"class.std::_String_val", ptr %4, i32 0, i32 0
  %10 = load ptr, ptr %9, align 8
  %11 = call noundef ptr @"??$_Unfancy@D@std@@YAPEADPEAD@Z"(ptr noundef %10) #3
  store ptr %11, ptr %3, align 8
  br label %12

12:                                               ; preds = %8, %1
  %13 = load ptr, ptr %3, align 8
  ret ptr %13
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"?move@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z"(ptr noundef %0, ptr noundef %1, i64 noundef %2) #0 comdat align 2 {
  %4 = alloca i64, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  store i64 %2, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store ptr %0, ptr %6, align 8
  %7 = load ptr, ptr %6, align 8
  %8 = load ptr, ptr %5, align 8
  %9 = load i64, ptr %4, align 8
  %10 = mul i64 %9, 1
  call void @llvm.memmove.p0.p0.i64(ptr align 1 %7, ptr align 1 %8, i64 %10, i1 false)
  %11 = load ptr, ptr %6, align 8
  ret ptr %11
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(32) ptr @"??$_Reallocate_grow_by@V<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@QEBD_K@Z@PEBD_K@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??append@01@QEAAAEAV01@QEBD0@Z@PEBD_K@Z"(ptr noundef nonnull align 8 dereferenceable(32) %0, i64 noundef %1, i8 %2, ptr noundef %3, i64 noundef %4) #10 comdat align 2 {
  %6 = alloca %class.anon.11, align 1
  %7 = alloca i64, align 8
  %8 = alloca ptr, align 8
  %9 = alloca i64, align 8
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca i64, align 8
  %13 = alloca i64, align 8
  %14 = alloca i64, align 8
  %15 = alloca i64, align 8
  %16 = alloca ptr, align 8
  %17 = alloca ptr, align 8
  %18 = alloca ptr, align 8
  %19 = alloca ptr, align 8
  %20 = getelementptr inbounds nuw %class.anon.11, ptr %6, i32 0, i32 0
  store i8 %2, ptr %20, align 1
  store i64 %4, ptr %7, align 8
  store ptr %3, ptr %8, align 8
  store i64 %1, ptr %9, align 8
  store ptr %0, ptr %10, align 8
  %21 = load ptr, ptr %10, align 8
  %22 = getelementptr inbounds nuw %"class.std::basic_string", ptr %21, i32 0, i32 0
  %23 = getelementptr inbounds nuw %"class.std::_Compressed_pair.10", ptr %22, i32 0, i32 0
  store ptr %23, ptr %11, align 8
  %24 = load ptr, ptr %11, align 8, !nonnull !16, !align !17
  %25 = getelementptr inbounds nuw %"class.std::_String_val", ptr %24, i32 0, i32 1
  %26 = load i64, ptr %25, align 8
  store i64 %26, ptr %12, align 8
  %27 = call noundef i64 @"?max_size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ"(ptr noundef nonnull align 8 dereferenceable(32) %21) #3
  %28 = load i64, ptr %12, align 8
  %29 = sub i64 %27, %28
  %30 = load i64, ptr %9, align 8
  %31 = icmp ult i64 %29, %30
  br i1 %31, label %32, label %33

32:                                               ; preds = %5
  call void @"?_Xlen_string@std@@YAXXZ"() #22
  unreachable

33:                                               ; preds = %5
  %34 = load i64, ptr %12, align 8
  %35 = load i64, ptr %9, align 8
  %36 = add i64 %34, %35
  store i64 %36, ptr %13, align 8
  %37 = load ptr, ptr %11, align 8, !nonnull !16, !align !17
  %38 = getelementptr inbounds nuw %"class.std::_String_val", ptr %37, i32 0, i32 2
  %39 = load i64, ptr %38, align 8
  store i64 %39, ptr %14, align 8
  %40 = load i64, ptr %13, align 8
  %41 = call noundef i64 @"?_Calculate_growth@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBA_K_K@Z"(ptr noundef nonnull align 8 dereferenceable(32) %21, i64 noundef %40) #3
  store i64 %41, ptr %15, align 8
  %42 = call noundef nonnull align 1 dereferenceable(1) ptr @"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV?$allocator@D@2@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %21) #3
  store ptr %42, ptr %16, align 8
  %43 = load ptr, ptr %16, align 8, !nonnull !16
  %44 = call noundef ptr @"??$_Allocate_for_capacity@$0A@@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@CAPEADAEAV?$allocator@D@1@AEA_K@Z"(ptr noundef nonnull align 1 dereferenceable(1) %43, ptr noundef nonnull align 8 dereferenceable(8) %15)
  store ptr %44, ptr %17, align 8
  %45 = load ptr, ptr %11, align 8, !nonnull !16, !align !17
  call void @"?_Orphan_all@_Container_base0@std@@QEAAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %45) #3
  %46 = load i64, ptr %13, align 8
  %47 = load ptr, ptr %11, align 8, !nonnull !16, !align !17
  %48 = getelementptr inbounds nuw %"class.std::_String_val", ptr %47, i32 0, i32 1
  store i64 %46, ptr %48, align 8
  %49 = load i64, ptr %15, align 8
  %50 = load ptr, ptr %11, align 8, !nonnull !16, !align !17
  %51 = getelementptr inbounds nuw %"class.std::_String_val", ptr %50, i32 0, i32 2
  store i64 %49, ptr %51, align 8
  %52 = load ptr, ptr %17, align 8
  %53 = call noundef ptr @"??$_Unfancy@D@std@@YAPEADPEAD@Z"(ptr noundef %52) #3
  store ptr %53, ptr %18, align 8
  %54 = load i64, ptr %14, align 8
  %55 = icmp ugt i64 %54, 15
  br i1 %55, label %56, label %72

56:                                               ; preds = %33
  %57 = load ptr, ptr %11, align 8, !nonnull !16, !align !17
  %58 = getelementptr inbounds nuw %"class.std::_String_val", ptr %57, i32 0, i32 0
  %59 = load ptr, ptr %58, align 8
  store ptr %59, ptr %19, align 8
  %60 = load i64, ptr %7, align 8
  %61 = load ptr, ptr %8, align 8
  %62 = load i64, ptr %12, align 8
  %63 = load ptr, ptr %19, align 8
  %64 = call noundef ptr @"??$_Unfancy@D@std@@YAPEADPEAD@Z"(ptr noundef %63) #3
  %65 = load ptr, ptr %18, align 8
  call void @"??R<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@QEBD_K@Z@QEBA?A?<auto>@@QEAD0101@Z"(ptr noundef nonnull align 1 dereferenceable(1) %6, ptr noundef %65, ptr noundef %64, i64 noundef %62, ptr noundef %61, i64 noundef %60)
  %66 = load i64, ptr %14, align 8
  %67 = load ptr, ptr %19, align 8
  %68 = load ptr, ptr %16, align 8, !nonnull !16
  call void @"?_Deallocate_for_capacity@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@CAXAEAV?$allocator@D@2@QEAD_K@Z"(ptr noundef nonnull align 1 dereferenceable(1) %68, ptr noundef %67, i64 noundef %66) #3
  %69 = load ptr, ptr %17, align 8
  %70 = load ptr, ptr %11, align 8, !nonnull !16, !align !17
  %71 = getelementptr inbounds nuw %"class.std::_String_val", ptr %70, i32 0, i32 0
  store ptr %69, ptr %71, align 8
  br label %82

72:                                               ; preds = %33
  %73 = load i64, ptr %7, align 8
  %74 = load ptr, ptr %8, align 8
  %75 = load i64, ptr %12, align 8
  %76 = load ptr, ptr %11, align 8, !nonnull !16, !align !17
  %77 = getelementptr inbounds nuw %"class.std::_String_val", ptr %76, i32 0, i32 0
  %78 = getelementptr inbounds [16 x i8], ptr %77, i64 0, i64 0
  %79 = load ptr, ptr %18, align 8
  call void @"??R<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@QEBD_K@Z@QEBA?A?<auto>@@QEAD0101@Z"(ptr noundef nonnull align 1 dereferenceable(1) %6, ptr noundef %79, ptr noundef %78, i64 noundef %75, ptr noundef %74, i64 noundef %73)
  %80 = load ptr, ptr %11, align 8, !nonnull !16, !align !17
  %81 = getelementptr inbounds nuw %"class.std::_String_val", ptr %80, i32 0, i32 0
  call void @"??$_Construct_in_place@PEADAEBQEAD@std@@YAXAEAPEADAEBQEAD@Z"(ptr noundef nonnull align 8 dereferenceable(8) %81, ptr noundef nonnull align 8 dereferenceable(8) %17) #3
  br label %82

82:                                               ; preds = %72, %56
  ret ptr %21
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef zeroext i1 @"?_Large_mode_engaged@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBA_NXZ"(ptr noundef nonnull align 8 dereferenceable(32) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"class.std::_String_val", ptr %3, i32 0, i32 2
  %5 = load i64, ptr %4, align 8
  %6 = icmp ugt i64 %5, 15
  ret i1 %6
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memmove.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg) #15

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?_Orphan_all@_Container_base0@std@@QEAAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"??R<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@QEBD_K@Z@QEBA?A?<auto>@@QEAD0101@Z"(ptr noundef nonnull align 1 dereferenceable(1) %0, ptr noundef %1, ptr noundef %2, i64 noundef %3, ptr noundef %4, i64 noundef %5) #0 comdat align 2 {
  %7 = alloca i64, align 8
  %8 = alloca ptr, align 8
  %9 = alloca i64, align 8
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca ptr, align 8
  %13 = alloca i8, align 1
  store i64 %5, ptr %7, align 8
  store ptr %4, ptr %8, align 8
  store i64 %3, ptr %9, align 8
  store ptr %2, ptr %10, align 8
  store ptr %1, ptr %11, align 8
  store ptr %0, ptr %12, align 8
  %14 = load ptr, ptr %12, align 8
  %15 = load i64, ptr %9, align 8
  %16 = load ptr, ptr %10, align 8
  %17 = load ptr, ptr %11, align 8
  %18 = call noundef ptr @"?copy@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z"(ptr noundef %17, ptr noundef %16, i64 noundef %15) #3
  %19 = load i64, ptr %7, align 8
  %20 = load ptr, ptr %8, align 8
  %21 = load ptr, ptr %11, align 8
  %22 = load i64, ptr %9, align 8
  %23 = getelementptr inbounds nuw i8, ptr %21, i64 %22
  %24 = call noundef ptr @"?copy@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z"(ptr noundef %23, ptr noundef %20, i64 noundef %19) #3
  store i8 0, ptr %13, align 1
  %25 = load ptr, ptr %11, align 8
  %26 = load i64, ptr %9, align 8
  %27 = load i64, ptr %7, align 8
  %28 = add i64 %26, %27
  %29 = getelementptr inbounds nuw i8, ptr %25, i64 %28
  call void @"?assign@?$_Narrow_char_traits@DH@std@@SAXAEADAEBD@Z"(ptr noundef nonnull align 1 dereferenceable(1) %29, ptr noundef nonnull align 1 dereferenceable(1) %13) #3
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?_Deallocate_for_capacity@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@CAXAEAV?$allocator@D@2@QEAD_K@Z"(ptr noundef nonnull align 1 dereferenceable(1) %0, ptr noundef %1, i64 noundef %2) #0 comdat align 2 {
  %4 = alloca i64, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  store i64 %2, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store ptr %0, ptr %6, align 8
  %7 = load ptr, ptr %6, align 8, !nonnull !16
  %8 = load i64, ptr %4, align 8
  %9 = add i64 %8, 1
  %10 = load ptr, ptr %5, align 8
  call void @"?deallocate@?$allocator@D@std@@QEAAXQEAD_K@Z"(ptr noundef nonnull align 1 dereferenceable(1) %7, ptr noundef %10, i64 noundef %9)
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?deallocate@?$allocator@D@std@@QEAAXQEAD_K@Z"(ptr noundef nonnull align 1 dereferenceable(1) %0, ptr noundef %1, i64 noundef %2) #0 comdat align 2 {
  %4 = alloca i64, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  store i64 %2, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store ptr %0, ptr %6, align 8
  %7 = load ptr, ptr %6, align 8
  %8 = load i64, ptr %4, align 8
  %9 = mul i64 1, %8
  %10 = load ptr, ptr %5, align 8
  call void @"??$_Deallocate@$0BA@$0A@@std@@YAXPEAX_K@Z"(ptr noundef %10, i64 noundef %9) #3
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"??$_Deallocate@$0BA@$0A@@std@@YAXPEAX_K@Z"(ptr noundef %0, i64 noundef %1) #0 comdat {
  %3 = alloca i64, align 8
  %4 = alloca ptr, align 8
  store i64 %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load i64, ptr %3, align 8
  %6 = icmp uge i64 %5, 4096
  br i1 %6, label %7, label %8

7:                                                ; preds = %2
  call void @"?_Adjust_manually_vector_aligned@std@@YAXAEAPEAXAEA_K@Z"(ptr noundef nonnull align 8 dereferenceable(8) %4, ptr noundef nonnull align 8 dereferenceable(8) %3)
  br label %8

8:                                                ; preds = %7, %2
  %9 = load i64, ptr %3, align 8
  %10 = load ptr, ptr %4, align 8
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %10, i64 noundef %9) #3
  ret void
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local void @"?_Adjust_manually_vector_aligned@std@@YAXAEAPEAXAEA_K@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(8) %1) #10 comdat {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i64, align 8
  %7 = alloca i64, align 8
  %8 = alloca i64, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %9 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %10 = load i64, ptr %9, align 8
  %11 = add i64 %10, 39
  store i64 %11, ptr %9, align 8
  %12 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %13 = load ptr, ptr %12, align 8
  store ptr %13, ptr %5, align 8
  %14 = load ptr, ptr %5, align 8
  %15 = getelementptr inbounds i64, ptr %14, i64 -1
  %16 = load i64, ptr %15, align 8
  store i64 %16, ptr %6, align 8
  store i64 8, ptr %7, align 8
  %17 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  %18 = load ptr, ptr %17, align 8
  %19 = ptrtoint ptr %18 to i64
  %20 = load i64, ptr %6, align 8
  %21 = sub i64 %19, %20
  store i64 %21, ptr %8, align 8
  br label %22

22:                                               ; preds = %2
  %23 = load i64, ptr %8, align 8
  %24 = icmp uge i64 %23, 8
  br i1 %24, label %25, label %29

25:                                               ; preds = %22
  %26 = load i64, ptr %8, align 8
  %27 = icmp ule i64 %26, 39
  br i1 %27, label %28, label %29

28:                                               ; preds = %25
  br label %32

29:                                               ; preds = %25, %22
  br label %30

30:                                               ; preds = %29
  call void @_invalid_parameter_noinfo_noreturn() #22
  unreachable

31:                                               ; No predecessors!
  br label %32

32:                                               ; preds = %31, %28
  br label %33

33:                                               ; preds = %32
  %34 = load i64, ptr %6, align 8
  %35 = inttoptr i64 %34 to ptr
  %36 = load ptr, ptr %4, align 8, !nonnull !16, !align !17
  store ptr %35, ptr %36, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBAPEBDXZ"(ptr noundef nonnull align 8 dereferenceable(32) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr inbounds nuw %"class.std::_String_val", ptr %4, i32 0, i32 0
  %6 = getelementptr inbounds [16 x i8], ptr %5, i64 0, i64 0
  store ptr %6, ptr %3, align 8
  %7 = call noundef zeroext i1 @"?_Large_mode_engaged@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBA_NXZ"(ptr noundef nonnull align 8 dereferenceable(32) %4) #3
  br i1 %7, label %8, label %12

8:                                                ; preds = %1
  %9 = getelementptr inbounds nuw %"class.std::_String_val", ptr %4, i32 0, i32 0
  %10 = load ptr, ptr %9, align 8
  %11 = call noundef ptr @"??$_Unfancy@D@std@@YAPEADPEAD@Z"(ptr noundef %10) #3
  store ptr %11, ptr %3, align 8
  br label %12

12:                                               ; preds = %8, %1
  %13 = load ptr, ptr %3, align 8
  ret ptr %13
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??$?0V?$allocator@D@std@@$$V@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@U_One_then_variadic_args_t@1@$$QEAV?$allocator@D@1@@Z"(ptr noundef nonnull returned align 8 dereferenceable(32) %0, i8 %1, ptr noundef nonnull align 1 dereferenceable(1) %2) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %4 = alloca %"struct.std::_One_then_variadic_args_t", align 1
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = getelementptr inbounds nuw %"struct.std::_One_then_variadic_args_t", ptr %4, i32 0, i32 0
  store i8 %1, ptr %7, align 1
  store ptr %2, ptr %5, align 8
  store ptr %0, ptr %6, align 8
  %8 = load ptr, ptr %6, align 8
  %9 = load ptr, ptr %5, align 8, !nonnull !16
  %10 = getelementptr inbounds nuw %"class.std::_Compressed_pair.10", ptr %8, i32 0, i32 0
  %11 = call noundef ptr @"??0?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %10) #3
  invoke void @llvm.seh.scope.begin()
          to label %12 unwind label %14

12:                                               ; preds = %3
  invoke void @llvm.seh.scope.end()
          to label %13 unwind label %14

13:                                               ; preds = %12
  ret ptr %8

14:                                               ; preds = %12, %3
  %15 = cleanuppad within none []
  call void @"??1?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %10) #3 [ "funclet"(token %15) ]
  cleanupret from %15 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?_Alloc_proxy@_Container_base0@std@@QEAAXAEBU_Fake_allocator@2@@Z"(ptr noundef nonnull align 1 dereferenceable(1) %0, ptr noundef nonnull align 1 dereferenceable(1) %1) #0 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?_Take_contents@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXAEAV12@@Z"(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(32) %1) #0 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %7 = load ptr, ptr %4, align 8
  %8 = getelementptr inbounds nuw %"class.std::basic_string", ptr %7, i32 0, i32 0
  %9 = getelementptr inbounds nuw %"class.std::_Compressed_pair.10", ptr %8, i32 0, i32 0
  store ptr %9, ptr %5, align 8
  %10 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %11 = getelementptr inbounds nuw %"class.std::basic_string", ptr %10, i32 0, i32 0
  %12 = getelementptr inbounds nuw %"class.std::_Compressed_pair.10", ptr %11, i32 0, i32 0
  store ptr %12, ptr %6, align 8
  %13 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  call void @"?_Memcpy_val_from@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXAEBV12@@Z"(ptr noundef nonnull align 8 dereferenceable(32) %7, ptr noundef nonnull align 8 dereferenceable(32) %13) #3
  %14 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  call void @"?_Tidy_init@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXXZ"(ptr noundef nonnull align 8 dereferenceable(32) %14) #3
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?_Memcpy_val_from@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXAEBV12@@Z"(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(32) %1) #0 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %7 = load ptr, ptr %4, align 8
  %8 = getelementptr inbounds nuw %"class.std::basic_string", ptr %7, i32 0, i32 0
  %9 = getelementptr inbounds nuw %"class.std::_Compressed_pair.10", ptr %8, i32 0, i32 0
  %10 = getelementptr inbounds nuw i8, ptr %9, i64 0
  store ptr %10, ptr %5, align 8
  %11 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %12 = getelementptr inbounds nuw %"class.std::basic_string", ptr %11, i32 0, i32 0
  %13 = getelementptr inbounds nuw %"class.std::_Compressed_pair.10", ptr %12, i32 0, i32 0
  %14 = getelementptr inbounds nuw i8, ptr %13, i64 0
  store ptr %14, ptr %6, align 8
  %15 = load ptr, ptr %5, align 8
  %16 = load ptr, ptr %6, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %15, ptr align 1 %16, i64 32, i1 false)
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?_Tidy_init@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXXZ"(ptr noundef nonnull align 8 dereferenceable(32) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca i8, align 1
  store ptr %0, ptr %2, align 8
  %5 = load ptr, ptr %2, align 8
  %6 = getelementptr inbounds nuw %"class.std::basic_string", ptr %5, i32 0, i32 0
  %7 = getelementptr inbounds nuw %"class.std::_Compressed_pair.10", ptr %6, i32 0, i32 0
  store ptr %7, ptr %3, align 8
  %8 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %9 = getelementptr inbounds nuw %"class.std::_String_val", ptr %8, i32 0, i32 1
  store i64 0, ptr %9, align 8
  %10 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %11 = getelementptr inbounds nuw %"class.std::_String_val", ptr %10, i32 0, i32 2
  store i64 15, ptr %11, align 8
  %12 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  call void @"?_Activate_SSO_buffer@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAAXXZ"(ptr noundef nonnull align 8 dereferenceable(32) %12) #3
  store i8 0, ptr %4, align 1
  %13 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %14 = getelementptr inbounds nuw %"class.std::_String_val", ptr %13, i32 0, i32 0
  %15 = getelementptr inbounds [16 x i8], ptr %14, i64 0, i64 0
  call void @"?assign@?$_Narrow_char_traits@DH@std@@SAXAEADAEBD@Z"(ptr noundef nonnull align 1 dereferenceable(1) %15, ptr noundef nonnull align 1 dereferenceable(1) %4) #3
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?_Activate_SSO_buffer@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAAXXZ"(ptr noundef nonnull align 8 dereferenceable(32) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?select_on_container_copy_construction@?$_Default_allocator_traits@V?$allocator@D@std@@@std@@SA?AV?$allocator@D@2@AEBV32@@Z"(ptr dead_on_unwind noalias writable sret(%"class.std::allocator") align 1 %0, ptr noundef nonnull align 1 dereferenceable(1) %1) #0 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8, !nonnull !16
  ret void
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define linkonce_odr dso_local void @"??$_Construct@$01PEBD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXQEBD_K@Z"(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef %1, i64 noundef %2) #10 comdat align 2 {
  %4 = alloca i64, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca %"struct.std::_Fake_proxy_ptr_impl", align 1
  %11 = alloca i64, align 8
  %12 = alloca ptr, align 8
  store i64 %2, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store ptr %0, ptr %6, align 8
  %13 = load ptr, ptr %6, align 8
  %14 = getelementptr inbounds nuw %"class.std::basic_string", ptr %13, i32 0, i32 0
  %15 = getelementptr inbounds nuw %"class.std::_Compressed_pair.10", ptr %14, i32 0, i32 0
  store ptr %15, ptr %7, align 8
  %16 = load i64, ptr %4, align 8
  %17 = call noundef i64 @"?max_size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ"(ptr noundef nonnull align 8 dereferenceable(32) %13) #3
  %18 = icmp ugt i64 %16, %17
  br i1 %18, label %19, label %20

19:                                               ; preds = %3
  call void @"?_Xlen_string@std@@YAXXZ"() #22
  unreachable

20:                                               ; preds = %3
  %21 = call noundef nonnull align 1 dereferenceable(1) ptr @"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV?$allocator@D@2@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %13) #3
  store ptr %21, ptr %8, align 8
  store ptr @"?_Fake_alloc@std@@3U_Fake_allocator@1@B", ptr %9, align 8
  %22 = load ptr, ptr %7, align 8, !nonnull !16, !align !17
  %23 = call noundef ptr @"??0_Fake_proxy_ptr_impl@std@@QEAA@AEBU_Fake_allocator@1@AEBU_Container_base0@1@@Z"(ptr noundef nonnull align 1 dereferenceable(1) %10, ptr noundef nonnull align 1 dereferenceable(1) @"?_Fake_alloc@std@@3U_Fake_allocator@1@B", ptr noundef nonnull align 1 dereferenceable(1) %22) #3
  %24 = load i64, ptr %4, align 8
  %25 = icmp ule i64 %24, 15
  br i1 %25, label %26, label %37

26:                                               ; preds = %20
  %27 = load i64, ptr %4, align 8
  %28 = load ptr, ptr %7, align 8, !nonnull !16, !align !17
  %29 = getelementptr inbounds nuw %"class.std::_String_val", ptr %28, i32 0, i32 1
  store i64 %27, ptr %29, align 8
  %30 = load ptr, ptr %7, align 8, !nonnull !16, !align !17
  %31 = getelementptr inbounds nuw %"class.std::_String_val", ptr %30, i32 0, i32 2
  store i64 15, ptr %31, align 8
  %32 = load ptr, ptr %5, align 8
  %33 = load ptr, ptr %7, align 8, !nonnull !16, !align !17
  %34 = getelementptr inbounds nuw %"class.std::_String_val", ptr %33, i32 0, i32 0
  %35 = getelementptr inbounds [16 x i8], ptr %34, i64 0, i64 0
  %36 = call noundef ptr @"?copy@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z"(ptr noundef %35, ptr noundef %32, i64 noundef 16) #3
  call void @"?_Release@_Fake_proxy_ptr_impl@std@@QEAAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %10) #3
  br label %58

37:                                               ; preds = %20
  %38 = load ptr, ptr %7, align 8, !nonnull !16, !align !17
  %39 = getelementptr inbounds nuw %"class.std::_String_val", ptr %38, i32 0, i32 2
  store i64 15, ptr %39, align 8
  %40 = load i64, ptr %4, align 8
  %41 = call noundef i64 @"?_Calculate_growth@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBA_K_K@Z"(ptr noundef nonnull align 8 dereferenceable(32) %13, i64 noundef %40) #3
  store i64 %41, ptr %11, align 8
  %42 = load ptr, ptr %8, align 8, !nonnull !16
  %43 = call noundef ptr @"??$_Allocate_for_capacity@$0A@@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@CAPEADAEAV?$allocator@D@1@AEA_K@Z"(ptr noundef nonnull align 1 dereferenceable(1) %42, ptr noundef nonnull align 8 dereferenceable(8) %11)
  store ptr %43, ptr %12, align 8
  %44 = load ptr, ptr %7, align 8, !nonnull !16, !align !17
  %45 = getelementptr inbounds nuw %"class.std::_String_val", ptr %44, i32 0, i32 0
  call void @"??$_Construct_in_place@PEADAEBQEAD@std@@YAXAEAPEADAEBQEAD@Z"(ptr noundef nonnull align 8 dereferenceable(8) %45, ptr noundef nonnull align 8 dereferenceable(8) %12) #3
  %46 = load i64, ptr %4, align 8
  %47 = load ptr, ptr %7, align 8, !nonnull !16, !align !17
  %48 = getelementptr inbounds nuw %"class.std::_String_val", ptr %47, i32 0, i32 1
  store i64 %46, ptr %48, align 8
  %49 = load i64, ptr %11, align 8
  %50 = load ptr, ptr %7, align 8, !nonnull !16, !align !17
  %51 = getelementptr inbounds nuw %"class.std::_String_val", ptr %50, i32 0, i32 2
  store i64 %49, ptr %51, align 8
  %52 = load i64, ptr %4, align 8
  %53 = add i64 %52, 1
  %54 = load ptr, ptr %5, align 8
  %55 = load ptr, ptr %12, align 8
  %56 = call noundef ptr @"??$_Unfancy@D@std@@YAPEADPEAD@Z"(ptr noundef %55) #3
  %57 = call noundef ptr @"?copy@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z"(ptr noundef %56, ptr noundef %54, i64 noundef %53) #3
  call void @"?_Release@_Fake_proxy_ptr_impl@std@@QEAAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %10) #3
  br label %58

58:                                               ; preds = %37, %26
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"?c_str@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBAPEBDXZ"(ptr noundef nonnull align 8 dereferenceable(32) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"class.std::basic_string", ptr %3, i32 0, i32 0
  %5 = getelementptr inbounds nuw %"class.std::_Compressed_pair.10", ptr %4, i32 0, i32 0
  %6 = call noundef ptr @"?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBAPEBDXZ"(ptr noundef nonnull align 8 dereferenceable(32) %5) #3
  ret ptr %6
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?_Tidy_deallocate@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXXZ"(ptr noundef nonnull align 8 dereferenceable(32) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i8, align 1
  store ptr %0, ptr %2, align 8
  %7 = load ptr, ptr %2, align 8
  %8 = getelementptr inbounds nuw %"class.std::basic_string", ptr %7, i32 0, i32 0
  %9 = getelementptr inbounds nuw %"class.std::_Compressed_pair.10", ptr %8, i32 0, i32 0
  store ptr %9, ptr %3, align 8
  %10 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  call void @"?_Orphan_all@_Container_base0@std@@QEAAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %10) #3
  %11 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %12 = call noundef zeroext i1 @"?_Large_mode_engaged@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBA_NXZ"(ptr noundef nonnull align 8 dereferenceable(32) %11) #3
  br i1 %12, label %13, label %26

13:                                               ; preds = %1
  %14 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %15 = getelementptr inbounds nuw %"class.std::_String_val", ptr %14, i32 0, i32 0
  %16 = load ptr, ptr %15, align 8
  store ptr %16, ptr %4, align 8
  %17 = call noundef nonnull align 1 dereferenceable(1) ptr @"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV?$allocator@D@2@XZ"(ptr noundef nonnull align 8 dereferenceable(32) %7) #3
  store ptr %17, ptr %5, align 8
  %18 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %19 = getelementptr inbounds nuw %"class.std::_String_val", ptr %18, i32 0, i32 0
  call void @"??$_Destroy_in_place@PEAD@std@@YAXAEAPEAD@Z"(ptr noundef nonnull align 8 dereferenceable(8) %19) #3
  %20 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  call void @"?_Activate_SSO_buffer@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAAXXZ"(ptr noundef nonnull align 8 dereferenceable(32) %20) #3
  %21 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %22 = getelementptr inbounds nuw %"class.std::_String_val", ptr %21, i32 0, i32 2
  %23 = load i64, ptr %22, align 8
  %24 = load ptr, ptr %4, align 8
  %25 = load ptr, ptr %5, align 8, !nonnull !16
  call void @"?_Deallocate_for_capacity@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@CAXAEAV?$allocator@D@2@QEAD_K@Z"(ptr noundef nonnull align 1 dereferenceable(1) %25, ptr noundef %24, i64 noundef %23) #3
  br label %26

26:                                               ; preds = %13, %1
  %27 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %28 = getelementptr inbounds nuw %"class.std::_String_val", ptr %27, i32 0, i32 1
  store i64 0, ptr %28, align 8
  %29 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %30 = getelementptr inbounds nuw %"class.std::_String_val", ptr %29, i32 0, i32 2
  store i64 15, ptr %30, align 8
  store i8 0, ptr %6, align 1
  %31 = load ptr, ptr %3, align 8, !nonnull !16, !align !17
  %32 = getelementptr inbounds nuw %"class.std::_String_val", ptr %31, i32 0, i32 0
  %33 = getelementptr inbounds [16 x i8], ptr %32, i64 0, i64 0
  call void @"?assign@?$_Narrow_char_traits@DH@std@@SAXAEADAEBD@Z"(ptr noundef nonnull align 1 dereferenceable(1) %33, ptr noundef nonnull align 1 dereferenceable(1) %6) #3
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"??$_Destroy_in_place@PEAD@std@@YAXAEAPEAD@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0) #0 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8, !nonnull !16, !align !17
  ret void
}

; Function Attrs: nounwind
declare dso_local noundef i32 @"?uncaught_exceptions@std@@YAHXZ"() #19

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  invoke void @llvm.seh.try.begin()
          to label %4 unwind label %44

4:                                                ; preds = %1
  %5 = getelementptr inbounds i8, ptr %3, i64 0
  %6 = load ptr, ptr %5, align 8
  %7 = getelementptr inbounds i32, ptr %6, i32 1
  %8 = load i32, ptr %7, align 4
  %9 = sext i32 %8 to i64
  %10 = add nsw i64 0, %9
  %11 = getelementptr inbounds i8, ptr %3, i64 %10
  %12 = call noundef zeroext i1 @"?good@ios_base@std@@QEBA_NXZ"(ptr noundef nonnull align 8 dereferenceable(72) %11) #3
  br i1 %12, label %13, label %52

13:                                               ; preds = %4
  %14 = getelementptr inbounds i8, ptr %3, i64 0
  %15 = load ptr, ptr %14, align 8
  %16 = getelementptr inbounds i32, ptr %15, i32 1
  %17 = load i32, ptr %16, align 4
  %18 = sext i32 %17 to i64
  %19 = add nsw i64 0, %18
  %20 = getelementptr inbounds i8, ptr %3, i64 %19
  %21 = call noundef i32 @"?flags@ios_base@std@@QEBAHXZ"(ptr noundef nonnull align 8 dereferenceable(72) %20) #3
  %22 = and i32 %21, 2
  %23 = icmp ne i32 %22, 0
  br i1 %23, label %24, label %52

24:                                               ; preds = %13
  %25 = getelementptr inbounds i8, ptr %3, i64 0
  %26 = load ptr, ptr %25, align 8
  %27 = getelementptr inbounds i32, ptr %26, i32 1
  %28 = load i32, ptr %27, align 4
  %29 = sext i32 %28 to i64
  %30 = add nsw i64 0, %29
  %31 = getelementptr inbounds i8, ptr %3, i64 %30
  %32 = call noundef ptr @"?rdbuf@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_streambuf@DU?$char_traits@D@std@@@2@XZ"(ptr noundef nonnull align 8 dereferenceable(96) %31) #3
  %33 = invoke noundef i32 @"?pubsync@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ"(ptr noundef nonnull align 8 dereferenceable(104) %32)
          to label %34 unwind label %44

34:                                               ; preds = %24
  %35 = icmp eq i32 %33, -1
  br i1 %35, label %36, label %51

36:                                               ; preds = %34
  %37 = getelementptr inbounds i8, ptr %3, i64 0
  %38 = load ptr, ptr %37, align 8
  %39 = getelementptr inbounds i32, ptr %38, i32 1
  %40 = load i32, ptr %39, align 4
  %41 = sext i32 %40 to i64
  %42 = add nsw i64 0, %41
  %43 = getelementptr inbounds i8, ptr %3, i64 %42
  invoke void @"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"(ptr noundef nonnull align 8 dereferenceable(96) %43, i32 noundef 4, i1 noundef zeroext false)
          to label %50 unwind label %44

44:                                               ; preds = %36, %24, %1
  %45 = catchswitch within none [label %46] unwind to caller

46:                                               ; preds = %44
  %47 = catchpad within %45 [ptr null, i32 0, ptr null]
  catchret from %47 to label %48

48:                                               ; preds = %46
  br label %49

49:                                               ; preds = %48, %52
  ret void

50:                                               ; preds = %36
  br label %51

51:                                               ; preds = %50, %34
  br label %52

52:                                               ; preds = %51, %13, %4
  br label %49
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??$end@D$0L@@std@@YAPEADAEAY0L@D@Z"(ptr noundef nonnull align 1 dereferenceable(11) %0) #0 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8, !nonnull !16
  %4 = getelementptr inbounds [11 x i8], ptr %3, i64 0, i64 0
  %5 = getelementptr inbounds nuw i8, ptr %4, i64 11
  ret ptr %5
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??$_UIntegral_to_buff@DI@std@@YAPEADPEADI@Z"(ptr noundef %0, i32 noundef %1) #0 comdat {
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  store i32 %1, ptr %3, align 4
  store ptr %0, ptr %4, align 8
  %6 = load i32, ptr %3, align 4
  store i32 %6, ptr %5, align 4
  br label %7

7:                                                ; preds = %16, %2
  %8 = load i32, ptr %5, align 4
  %9 = urem i32 %8, 10
  %10 = add i32 48, %9
  %11 = trunc i32 %10 to i8
  %12 = load ptr, ptr %4, align 8
  %13 = getelementptr inbounds i8, ptr %12, i32 -1
  store ptr %13, ptr %4, align 8
  store i8 %11, ptr %13, align 1
  %14 = load i32, ptr %5, align 4
  %15 = udiv i32 %14, 10
  store i32 %15, ptr %5, align 4
  br label %16

16:                                               ; preds = %7
  %17 = load i32, ptr %5, align 4
  %18 = icmp ne i32 %17, 0
  br i1 %18, label %7, label %19, !llvm.loop !27

19:                                               ; preds = %16
  %20 = load ptr, ptr %4, align 8
  ret ptr %20
}

attributes #0 = { mustprogress noinline nounwind optnone sspstrong uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress noinline optnone presplitcoroutine sspstrong uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #3 = { nounwind }
attributes #4 = { nobuiltin allocsize(0) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { nounwind memory(none) }
attributes #6 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #7 = { nomerge nounwind }
attributes #8 = { alwaysinline mustprogress "min-legal-vector-width"="0" }
attributes #9 = { nounwind willreturn memory(write) }
attributes #10 = { mustprogress noinline optnone sspstrong uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #11 = { nobuiltin nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #12 = { nounwind memory(argmem: read) }
attributes #13 = { mustprogress noinline norecurse optnone sspstrong uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #14 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #15 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #16 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #17 = { noreturn "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #18 = { noreturn nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #19 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #20 = { mustprogress noinline noreturn optnone sspstrong uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #21 = { allocsize(0) }
attributes #22 = { noreturn }
attributes #23 = { builtin nounwind }
attributes #24 = { builtin allocsize(0) }
attributes #25 = { noreturn nounwind }

!llvm.linker.options = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9}
!llvm.module.flags = !{!10, !11, !12, !13, !14}
!llvm.ident = !{!15}

!0 = !{!"/DEFAULTLIB:libcmt.lib"}
!1 = !{!"/DEFAULTLIB:oldnames.lib"}
!2 = !{!"/FAILIFMISMATCH:\22_COROUTINE_ABI=2\22"}
!3 = !{!"/FAILIFMISMATCH:\22_MSC_VER=1900\22"}
!4 = !{!"/FAILIFMISMATCH:\22_ITERATOR_DEBUG_LEVEL=0\22"}
!5 = !{!"/FAILIFMISMATCH:\22RuntimeLibrary=MT_StaticRelease\22"}
!6 = !{!"/DEFAULTLIB:libcpmt.lib"}
!7 = !{!"/FAILIFMISMATCH:\22_CRT_STDIO_ISO_WIDE_SPECIFIERS=0\22"}
!8 = !{!"/FAILIFMISMATCH:\22annotate_string=0\22"}
!9 = !{!"/FAILIFMISMATCH:\22annotate_vector=0\22"}
!10 = !{i32 1, !"wchar_size", i32 2}
!11 = !{i32 2, !"eh-asynch", i32 1}
!12 = !{i32 8, !"PIC Level", i32 2}
!13 = !{i32 7, !"uwtable", i32 2}
!14 = !{i32 1, !"MaxTLSAlign", i32 65536}
!15 = !{!"clang version 21.0.0git (https://github.com/llvm/llvm-project.git e66c205bda33a91fbe2ba5b4a5d6b823e5c23e8a)"}
!16 = !{}
!17 = !{i64 8}
!18 = distinct !{!18, !19}
!19 = !{!"llvm.loop.mustprogress"}
!20 = distinct !{!20, !19}
!21 = distinct !{!21, !19}
!22 = distinct !{!22, !19}
!23 = !{i64 4}
!24 = distinct !{!24, !19}
!25 = distinct !{!25, !19}
!26 = distinct !{!26, !19}
!27 = distinct !{!27, !19}
