// REQUIRES: x86_64-unknown-linux-gnu
// RUN: %offload-tblgen -gen-acc-c-enums -I %S/../../../libacctarget/API %s | %fcheck-generic --check-prefix=CENUMS
// RUN: %offload-tblgen -gen-acc-exports -I %S/../../../libacctarget/API %s | %fcheck-generic --check-prefix=EXPORT
// RUN: %offload-tblgen -gen-acc-f-module -I %S/../../../libacctarget/API %s | %fcheck-generic --check-prefix=MOD
// RUN: %offload-tblgen -gen-acc-print-enums -I %S/../../../libacctarget/API %s | %fcheck-generic --check-prefix=PRINT
// RUN: %offload-tblgen -gen-acc-f-entries -I %S/../../../libacctarget/API %s | %fcheck-generic --check-prefix=ENTRY
// RUN: %offload-tblgen -gen-acc-f-impl-decls -I %S/../../../libacctarget/API %s | %fcheck-generic --check-prefix=DECL
// RUN: %offload-tblgen -gen-acc-f-impls -I %S/../../../libacctarget/API %s | %fcheck-generic --check-prefix=IMPL

include "AccAPI.td"

// CENUMS:      #define acc_async_sync (-1)
// CENUMS-NEXT: #define acc_async_default (-3)
// CENUMS-NEXT: #define acc_async_noval (-4)

// CENUMS:      typedef enum acc_device_property_t {
// CENUMS-NEXT:   acc_property_memory = 1,
// CENUMS-NEXT:   acc_property_free_memory = 2,

// CENUMS:        acc_property_shared_memory_support = 6
// CENUMS-NEXT: } acc_device_property_t;

// EXPORT:      VERS1.0 {
// EXPORT-NEXT:   global:
// EXPORT-NEXT:     __olacc_f_acc_async_test;

// EXPORT:          __tgt_acc_rtl_init;

// EXPORT:        local:
// EXPORT-NEXT:     *;
// EXPORT-NEXT: };

// MOD:        integer, parameter :: openacc_version = 202506
// MOD-NEXT:   integer, parameter :: acc_device_kind = c_int
// MOD-NEXT:   integer, parameter :: acc_device_property_kind = c_int
// MOD-NEXT:   integer, parameter :: acc_handle_kind = 8

// MOD:        integer(acc_device_kind), parameter :: acc_device_none = 0
// MOD-NEXT:   integer(acc_device_kind), parameter :: acc_device_default = 1

// MOD:        integer(acc_handle_kind), parameter :: acc_async_sync = -1
// MOD-NEXT:   integer(acc_handle_kind), parameter :: acc_async_default = -3
// MOD-NEXT:   integer(acc_handle_kind), parameter :: acc_async_noval = -4

// MOD:        interface acc_async_test
// MOD-NEXT:     logical(c_bool) function acc_async_test(wait_arg) bind(c, name="__olacc_f_acc_async_test")
// MOD-NEXT:       import :: c_bool, acc_handle_kind

// MOD:        interface acc_copyin
// MOD-NEXT:     subroutine acc_copyin(data_arg, bytes) bind(c, name="__olacc_f_acc_copyin")
// MOD-NEXT:       import :: c_size_t
// MOD-NEXT:       type(*), dimension(*) :: data_arg
// MOD-NEXT:       !dir$ ignore_tkr (k) bytes
// MOD-NEXT:       integer(c_size_t), value :: bytes
// MOD-NEXT:     end subroutine
// MOD-EMPTY:
// MOD-NEXT:     subroutine cfi_acc_copyin(data_arg) bind(c, name="__olacc_f_cfi_acc_copyin")
// MOD-NEXT:       type(*), dimension(..) :: data_arg
// MOD-NEXT:     end subroutine
// MOD-EMPTY:
// MOD-NEXT:   end interface
// MOD-EMPTY:
// MOD-NEXT:   interface acc_pcopyin
// MOD-NEXT:     procedure :: acc_copyin, cfi_acc_copyin
// MOD-NEXT:   end interface

// MOD:        interface acc_get_device_type
// MOD-NEXT:     integer(acc_device_kind) function acc_get_device_type() bind(c, name="__olacc_f_acc_get_device_type")
// MOD-NEXT:       import :: acc_device_kind
// MOD-NEXT:     end function
// MOD-EMPTY:
// MOD-NEXT:   end interface

// MOD:        interface acc_get_property_string
// MOD-NEXT:     subroutine acc_get_property_string(dev_num, dev_type, property, string) bind(c, name="__olacc_f_acc_get_property_string")
// MOD-NEXT:       import :: c_int, acc_device_kind, acc_device_property_kind

// MOD:            character(len=*), intent(out) :: string
// MOD-NEXT:     end subroutine

// MOD:          subroutine acc_wait_device_async(wait_arg, async_arg, dev_num) bind(c, name="__olacc_f_acc_wait_device_async")
// MOD-NEXT:       import :: acc_handle_kind, c_int
// MOD-NEXT:       !dir$ ignore_tkr (k) wait_arg

// PRINT:      {{\[\[}}maybe_unused{{\]\]}} static const char *enumName(acc_device_property_t V) {
// PRINT-NEXT:   switch (V) {
// PRINT-NEXT:   case acc_property_memory:
// PRINT-NEXT:     return "acc_property_memory";

// PRINT:        case acc_property_shared_memory_support:
// PRINT-NEXT:     return "acc_property_shared_memory_support";
// PRINT-NEXT:   }
// PRINT-NEXT:   return nullptr;
// PRINT-NEXT: }

// PRINT:      {{\[\[}}maybe_unused{{\]\]}} static const char *enumTypeName(acc_device_property_t) {
// PRINT-NEXT:   return "acc_device_property_t";
// PRINT-NEXT: }

// PRINT:      {{\[\[}}maybe_unused{{\]\]}} static const char *asyncConstName(int64_t V) {
// PRINT-NEXT:   switch (V) {
// PRINT-NEXT:   case acc_async_sync:
// PRINT-NEXT:     return "acc_async_sync";
// PRINT-NEXT:   case acc_async_default:
// PRINT-NEXT:     return "acc_async_default";
// PRINT-NEXT:   case acc_async_noval:
// PRINT-NEXT:     return "acc_async_noval";
// PRINT-NEXT:   }
// PRINT-NEXT:   return nullptr;
// PRINT-NEXT: }

// ENTRY:      ACC_API_EXPORT bool __olacc_f_acc_async_test(int64_t wait_arg) {
// ENTRY-NEXT:   ODBG(ADT_APITrace) << "enter acc_async_test(" << llvm::acc::print::async(wait_arg) << ")";
// ENTRY-NEXT:   bool res = llvm::acc::impl::f::acc_async_test(wait_arg);
// ENTRY-NEXT:   ODBG(ADT_APITrace) << "exit  acc_async_test -> " << llvm::acc::print::logical(res);
// ENTRY-NEXT:   return res;
// ENTRY-NEXT: }

// ENTRY:      ACC_API_EXPORT acc_device_t __olacc_f_acc_get_device_type(void) {
// ENTRY-NEXT:   ODBG(ADT_APITrace) << "enter acc_get_device_type(" << ")";
// ENTRY-NEXT:   acc_device_t res = llvm::acc::impl::f::acc_get_device_type();
// ENTRY-NEXT:   ODBG(ADT_APITrace) << "exit  acc_get_device_type -> " << res;
// ENTRY-NEXT:   return res;
// ENTRY-NEXT: }

// ENTRY:      ACC_API_EXPORT void __olacc_f_acc_get_property_string(int dev_num, acc_device_t dev_type, acc_device_property_t property, CFI_cdesc_t *string) {
// ENTRY-NEXT:   ODBG(ADT_APITrace) << "enter acc_get_property_string(" << dev_num << ", " << dev_type << ", " << property << ", " << string << ")";
// ENTRY-NEXT:   llvm::acc::impl::f::acc_get_property_string(dev_num, dev_type, property, string);
// ENTRY-NEXT:   ODBG(ADT_APITrace) << "exit  acc_get_property_string";
// ENTRY-NEXT: }

// ENTRY:      ACC_API_EXPORT void __olacc_f_acc_memcpy_d2d(void *data_arg_dest, void *data_arg_src, size_t bytes, int dev_num_dest, int dev_num_src) {
// ENTRY-NEXT:   ODBG(ADT_APITrace) << "enter acc_memcpy_d2d(" << data_arg_dest << ", " << data_arg_src << ", " << bytes << ", " << dev_num_dest << ", " << dev_num_src << ")";
// ENTRY-NEXT:   llvm::acc::impl::f::acc_memcpy_d2d(data_arg_dest, data_arg_src, bytes, dev_num_dest, dev_num_src);
// ENTRY-NEXT:   ODBG(ADT_APITrace) << "exit  acc_memcpy_d2d";
// ENTRY-NEXT: }

// DECL:      bool acc_async_test(int64_t wait_arg);
// DECL:      void *acc_deviceptr(void *data_arg);
// DECL:      acc_device_t acc_get_device_type(void);
// DECL:      void acc_wait(int64_t wait_arg);

// IMPL:      acc_device_t acc_get_device_type(void) {
// IMPL-NEXT:   ACC_API_NYI();
// IMPL-NEXT: }

// IMPL:      void acc_wait(int64_t wait_arg) {
// IMPL-NEXT:   ACC_API_NYI();
// IMPL-NEXT: }
