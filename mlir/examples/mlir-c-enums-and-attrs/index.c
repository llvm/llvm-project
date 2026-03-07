#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

#include <mlir-c/IR.h>
#include <mlir-c/Support.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/Dialect/Func.h>
#include <mlir-c/Dialect/Index.h>


// HACK
typedef enum indexIndexCmpPredicate indexIndexCmpPredicate;

/* forward decls */

// Simple callback for MLIR *Print routines. It just prints to the stdout.
void print_callback(MlirStringRef r, void *data);
// Load dialect and error check
MlirDialect load_dialect(MlirContext ctx, MlirDialectHandle handle);

#define HERE(ctx) (mlirLocationFileLineColGet(ctx, mlirStringRefCreateFromCString(__FILE__),\
                                                   __LINE__, 0))

int main(int argc, char **argv)
{
    // Initialize the MLIR context 
    MlirContext ctx = mlirContextCreate();

    // Load the HW and SV Dialects
    MlirDialect func = load_dialect(ctx, mlirGetDialectHandle__func__());
    MlirDialect index = load_dialect(ctx, mlirGetDialectHandle__index__());

    MlirBlock top_block = mlirBlockCreate(0, NULL, NULL);

    MlirType index_type = mlirIndexTypeGet(ctx);
    MlirBlock func_body = mlirBlockCreate(0, NULL, NULL);
    MlirRegion func_region = mlirRegionCreate();
    mlirRegionAppendOwnedBlock(func_region, func_body);

    // Add two arguments, for the a and b parameters
    MlirValue arg_a = mlirBlockAddArgument(func_body, index_type, HERE(ctx));
    MlirValue arg_b = mlirBlockAddArgument(func_body, index_type, HERE(ctx));
    MlirValue cmp_args[] = {arg_a,arg_b};

    MlirAttribute cmp_attr = indexIndexCmpPredicateAttrGet(ctx, EQ);

    MlirNamedAttribute cmp_pred 
            = mlirNamedAttributeGet(mlirIdentifierGet(ctx,mlirStringRefCreateFromCString("pred")),
                                    cmp_attr);
    MlirNamedAttribute cmp_attrs[] = {cmp_pred};

    MlirType cmp_result_type = mlirIntegerTypeGet(ctx,1);

    MlirOperationState cmp_op_state 
                = mlirOperationStateGet(mlirStringRefCreateFromCString("index.cmp"), HERE(ctx));
    mlirOperationStateAddOperands(&cmp_op_state, 2, cmp_args);
    mlirOperationStateAddResults(&cmp_op_state, 1, &cmp_result_type);
    mlirOperationStateAddAttributes(&cmp_op_state, 1, cmp_attrs);
    MlirOperation cmp_op = mlirOperationCreate(&cmp_op_state);
    if (mlirOperationIsNull(cmp_op)) {
        printf("cmp_op is null");
        exit(-1);
    }
    mlirBlockAppendOwnedOperation(func_body, cmp_op);

    MlirOperationState return_op_state 
                = mlirOperationStateGet(mlirStringRefCreateFromCString("func.return"), HERE(ctx));
    MlirOperation return_op = mlirOperationCreate(&return_op_state);
    mlirBlockAppendOwnedOperation(func_body, return_op);

    MlirOperationState func_state 
                = mlirOperationStateGet(mlirStringRefCreateFromCString("func.func"), HERE(ctx));
    // mlirOperationStateAddOperands(&func_state, 2, cmp_args);
    MlirAttribute func_sym_attr = mlirStringAttrGet(ctx,mlirStringRefCreateFromCString("cmp_op"));
    MlirNamedAttribute func_named_sym_attr = mlirNamedAttributeGet(
        mlirIdentifierGet(ctx,mlirStringRefCreateFromCString("sym_name")),
        func_sym_attr);
    
    
    MlirType func_input_types[] = {index_type, index_type};
    MlirType func_type = mlirFunctionTypeGet(ctx, 2, func_input_types, 0, NULL);
    MlirAttribute func_type_attr = mlirTypeAttrGet(func_type);

    MlirNamedAttribute func_named_type_attr = mlirNamedAttributeGet(
        mlirIdentifierGet(ctx,mlirStringRefCreateFromCString("function_type")), 
        func_type_attr);
    MlirNamedAttribute func_attrs[] = {func_named_sym_attr, func_named_type_attr};
    mlirOperationStateAddAttributes(&func_state, 2, func_attrs);
    MlirRegion func_regions[] = {func_region};
    mlirOperationStateAddOwnedRegions(&func_state, 1, func_regions);

    

    MlirOperation func_op = mlirOperationCreate(&func_state);
    if (mlirOperationIsNull(func_op)) {
        printf("func_op is null");
        exit(-1);
    }

    MlirModule top = mlirModuleFromOperation(func_op);
    
    if (!mlirOperationVerify(func_op)) {
        printf("func_op failed verification");
        exit(-1);
    }
    mlirOperationDump(func_op);
    exit(0);
}

MlirDialect load_dialect(MlirContext ctx, MlirDialectHandle handle)
{
    return mlirDialectHandleLoadDialect(handle, ctx);
}

// Simple callback for MLIR *Print routines. It just prints to the stdout.
void print_callback(MlirStringRef r, void *data)
{
    write(0, r.data, r.length);
}
