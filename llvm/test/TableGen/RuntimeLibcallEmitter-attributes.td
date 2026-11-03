// RUN: llvm-tblgen -gen-runtime-libcalls -I %p/../../include %s | FileCheck %s

include "llvm/IR/RuntimeLibcallsImpl.td"

def SIN_F32 : RuntimeLibcall<Flt, [Flt]>;
def SINCOS_V4F32 : RuntimeLibcall<Void, [V4F32, Ptr, Ptr]> {
  let FnAttrs = [WillReturn, NoUnwind];
  let Memory = ArgMemWriteOnly;
  let ParamAttrs = [[], [NoAlias, NonNull], [NoAlias, NonNull]];
  let ParamPointeeTypes = [NoneType, Flt, Flt];
}

def MODF_V4F32 : RuntimeLibcall<V4F32, [V4F32, Ptr]> {
  let FnAttrs = [NoUnwind, WillReturn];
  let ParamAttrs = [[], [NonNull, NoAlias]];
}

def sinf : RuntimeLibcallImpl<SIN_F32>;
def vsincosf : RuntimeLibcallImpl<SINCOS_V4F32>;
def vsincosf_alt : RuntimeLibcallImpl<SINCOS_V4F32>;
def vmodff : RuntimeLibcallImpl<MODF_V4F32>;

// CHECK-LABEL: #ifdef GET_RUNTIME_LIBCALL_ATTRIBUTES
// CHECK-NEXT: #undef GET_RUNTIME_LIBCALL_ATTRIBUTES
// CHECK-EMPTY:
// CHECK-NEXT: static AttributeSet getLibcallFnAttributeSet(LLVMContext &C, unsigned ID) {
// CHECK-NEXT:   switch (ID) {
// CHECK-NEXT:   default:
// CHECK-NEXT:     llvm_unreachable("invalid libcall attribute set");
// CHECK-NEXT:   case 0:
// CHECK-NEXT:     return AttributeSet();
// CHECK-NEXT:   case 1:
// CHECK-NEXT:     return AttributeSet::get(C, {
// CHECK-NEXT:       Attribute::get(C, Attribute::NoUnwind),
// CHECK-NEXT:       Attribute::get(C, Attribute::WillReturn),
// CHECK-NEXT:     });
// CHECK-NEXT:   case 2:
// CHECK-NEXT:     return AttributeSet::get(C, {
// CHECK-NEXT:       Attribute::get(C, Attribute::NoUnwind),
// CHECK-NEXT:       Attribute::get(C, Attribute::WillReturn),
// CHECK-NEXT:       Attribute::getWithMemoryEffects(C, MemoryEffects::createFromIntValue(2)),
// CHECK-NEXT:     });
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK-NEXT: static AttributeSet getLibcallArgAttributeSet(LLVMContext &C, unsigned ID) {
// CHECK-NEXT:   switch (ID) {
// CHECK-NEXT:   default:
// CHECK-NEXT:     llvm_unreachable("invalid libcall attribute set");
// CHECK-NEXT:   case 0:
// CHECK-NEXT:     return AttributeSet::get(C, {
// CHECK-NEXT:       Attribute::get(C, Attribute::NoAlias),
// CHECK-NEXT:       Attribute::get(C, Attribute::NonNull),
// CHECK-NEXT:     });
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK-NEXT: namespace {
// CHECK-NEXT: struct LibcallArgAttrs {
// CHECK-NEXT:   uint8_t ArgNo;
// CHECK-NEXT:   uint8_t AttrSetID;
// CHECK-NEXT:   RTLIB::FuncArgTypeID PointeeType;
// CHECK-NEXT: };
// CHECK-NEXT: } // namespace
// CHECK-EMPTY:
// CHECK-NEXT: static constexpr LibcallArgAttrs LibcallArgAttrTable[] = {
// CHECK-NEXT:   /* 0 */ {1, 0, RTLIB::NoFuncArgType},
// CHECK-NEXT:   /* 1 */ {1, 0, RTLIB::Flt}, {2, 0, RTLIB::Flt},
// CHECK-NEXT: };
// CHECK-EMPTY:
// CHECK-NEXT: static constexpr struct {
// CHECK-NEXT:   uint8_t Start;
// CHECK-NEXT:   uint8_t Size;
// CHECK-NEXT: } LibcallArgAttrSeqs[] = {
// CHECK-NEXT:   {0, 0},
// CHECK-NEXT:   {0, 1},
// CHECK-NEXT:   {1, 2},
// CHECK-NEXT: };
// CHECK-EMPTY:
// CHECK-NEXT: static constexpr struct {
// CHECK-NEXT:   uint8_t FnAttrSetID;
// CHECK-NEXT:   uint8_t ArgAttrSeqID;
// CHECK-NEXT: } LibcallImplAttrIDs[] = {
// CHECK-NEXT:   {0, 0}, // RTLIB::Unsupported
// CHECK-NEXT:   {0, 0}, // RTLIB::impl_sinf
// CHECK-NEXT:   {1, 1}, // RTLIB::impl_vmodff
// CHECK-NEXT:   {2, 2}, // RTLIB::impl_vsincosf
// CHECK-NEXT:   {2, 2}, // RTLIB::impl_vsincosf_alt
// CHECK-NEXT: };
// CHECK-EMPTY:
// CHECK-NEXT: static AttributeSet getLibcallImplFnAttributes(LLVMContext &C,
// CHECK-NEXT:     RTLIB::LibcallImpl Impl) {
// CHECK-NEXT:   return getLibcallFnAttributeSet(C, LibcallImplAttrIDs[Impl].FnAttrSetID);
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK-NEXT: static ArrayRef<LibcallArgAttrs>
// CHECK-NEXT: getLibcallImplArgAttributes(RTLIB::LibcallImpl Impl) {
// CHECK-NEXT:   auto [Start, Size] =
// CHECK-NEXT:       LibcallArgAttrSeqs[LibcallImplAttrIDs[Impl].ArgAttrSeqID];
// CHECK-NEXT:   return ArrayRef(LibcallArgAttrTable).slice(Start, Size);
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK-NEXT: #endif // GET_RUNTIME_LIBCALL_ATTRIBUTES
