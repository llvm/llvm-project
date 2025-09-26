// RUN: %offload-tblgen -gen-entry-points -I %S/../../../liboffload/API %s | %fcheck-generic

// Check entry point wrapper functions are generated correctly

include "APIDefs.td"

def FunctionA : Function {
    let desc = "Function A description";
    let details = [ "Function A detailed information" ];
    let params = [
        Param<"uint32_t", "ParamA", "Parameter A description">,
        Param<"uint32_t*", "ParamB", "Parameter B description">,
    ];
    let returns = [
        Return<"OL_ERRC_INVALID_VALUE", ["When a value is invalid"]>
    ];
}


// The validation function should call the implementation function
// CHECK: FunctionA_val
// CHECK: return llvm::offload::FunctionA_impl(ParamA, ParamB);

// CHECK: ol_result_t{{.*}} FunctionA(

// The entry point should print tracing output if enabled
// CHECK: if (llvm::offload::isTracingEnabled()) {
// CHECK-NEXT: "---> FunctionA";

// CHECK: Result = llvmErrorToOffloadError(FunctionA_val(ParamA, ParamB));

// Tracing should construct a param struct for printing
// CHECK: if (llvm::offload::isTracingEnabled()) {
// CHECK: function_a_params_t Params = {&ParamA, &ParamB};

// CHECK: return Result;
