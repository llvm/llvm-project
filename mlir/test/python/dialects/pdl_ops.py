# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects.pdl import *


def constructAndPrintInModule(f):
  print("\nTEST:", f.__name__)
  with Context(), Location.unknown():
    module = Module.create()
    with InsertionPoint(module.body):
      f()
    print(module)
  return f


# CHECK: module  {
# CHECK:   pdl.pattern @operations : benefit(1)  {
# CHECK:     %0 = attribute
# CHECK:     %1 = type
# CHECK:     %2 = operation  {"attr" = %0} -> (%1 : !pdl.type)
# CHECK:     %3 = result 0 of %2
# CHECK:     %4 = operand
# CHECK:     %5 = operation(%3, %4 : !pdl.value, !pdl.value)
# CHECK:     rewrite %5 with "rewriter"
# CHECK:   }
# CHECK: }
@constructAndPrintInModule
def test_operations():
  pattern = PatternOp(1, "operations")
  with InsertionPoint(pattern.body):
    attr = AttributeOp()
    ty = TypeOp()
    op0 = OperationOp(attributes={"attr": attr}, types=[ty])
    op0_result = ResultOp(op0, 0)
    input = OperandOp()
    root = OperationOp(args=[op0_result, input])
    RewriteOp(root, "rewriter")


# CHECK: module  {
# CHECK:   pdl.pattern @rewrite_with_args : benefit(1)  {
# CHECK:     %0 = operand
# CHECK:     %1 = operation(%0 : !pdl.value)
# CHECK:     rewrite %1 with "rewriter"(%0 : !pdl.value)
# CHECK:   }
# CHECK: }
@constructAndPrintInModule
def test_rewrite_with_args():
  pattern = PatternOp(1, "rewrite_with_args")
  with InsertionPoint(pattern.body):
    input = OperandOp()
    root = OperationOp(args=[input])
    RewriteOp(root, "rewriter", args=[input])

# CHECK: module  {
# CHECK:   pdl.pattern @rewrite_multi_root_optimal : benefit(1)  {
# CHECK:     %0 = operand
# CHECK:     %1 = operand
# CHECK:     %2 = type
# CHECK:     %3 = operation(%0 : !pdl.value)  -> (%2 : !pdl.type)
# CHECK:     %4 = result 0 of %3
# CHECK:     %5 = operation(%4 : !pdl.value)
# CHECK:     %6 = operation(%1 : !pdl.value)  -> (%2 : !pdl.type)
# CHECK:     %7 = result 0 of %6
# CHECK:     %8 = operation(%4, %7 : !pdl.value, !pdl.value)
# CHECK:     rewrite with "rewriter"(%5, %8 : !pdl.operation, !pdl.operation)
# CHECK:   }
# CHECK: }
@constructAndPrintInModule
def test_rewrite_multi_root_optimal():
  pattern = PatternOp(1, "rewrite_multi_root_optimal")
  with InsertionPoint(pattern.body):
    input1 = OperandOp()
    input2 = OperandOp()
    ty = TypeOp()
    op1 = OperationOp(args=[input1], types=[ty])
    val1 = ResultOp(op1, 0)
    root1 = OperationOp(args=[val1])
    op2 = OperationOp(args=[input2], types=[ty])
    val2 = ResultOp(op2, 0)
    root2 = OperationOp(args=[val1, val2])
    RewriteOp(name="rewriter", args=[root1, root2])

# CHECK: module  {
# CHECK:   pdl.pattern @rewrite_multi_root_forced : benefit(1)  {
# CHECK:     %0 = operand
# CHECK:     %1 = operand
# CHECK:     %2 = type
# CHECK:     %3 = operation(%0 : !pdl.value)  -> (%2 : !pdl.type)
# CHECK:     %4 = result 0 of %3
# CHECK:     %5 = operation(%4 : !pdl.value)
# CHECK:     %6 = operation(%1 : !pdl.value)  -> (%2 : !pdl.type)
# CHECK:     %7 = result 0 of %6
# CHECK:     %8 = operation(%4, %7 : !pdl.value, !pdl.value)
# CHECK:     rewrite %5 with "rewriter"(%8 : !pdl.operation)
# CHECK:   }
# CHECK: }
@constructAndPrintInModule
def test_rewrite_multi_root_forced():
  pattern = PatternOp(1, "rewrite_multi_root_forced")
  with InsertionPoint(pattern.body):
    input1 = OperandOp()
    input2 = OperandOp()
    ty = TypeOp()
    op1 = OperationOp(args=[input1], types=[ty])
    val1 = ResultOp(op1, 0)
    root1 = OperationOp(args=[val1])
    op2 = OperationOp(args=[input2], types=[ty])
    val2 = ResultOp(op2, 0)
    root2 = OperationOp(args=[val1, val2])
    RewriteOp(root1, name="rewriter", args=[root2])

# CHECK: module  {
# CHECK:   pdl.pattern @rewrite_add_body : benefit(1)  {
# CHECK:     %0 = type : i32
# CHECK:     %1 = type
# CHECK:     %2 = operation  -> (%0, %1 : !pdl.type, !pdl.type)
# CHECK:     rewrite %2  {
# CHECK:       %3 = type
# CHECK:       %4 = operation "foo.op"  -> (%0, %3 : !pdl.type, !pdl.type)
# CHECK:       replace %2 with %4
# CHECK:     }
# CHECK:   }
# CHECK: }
@constructAndPrintInModule
def test_rewrite_add_body():
  pattern = PatternOp(1, "rewrite_add_body")
  with InsertionPoint(pattern.body):
    ty1 = TypeOp(IntegerType.get_signless(32))
    ty2 = TypeOp()
    root = OperationOp(types=[ty1, ty2])
    rewrite = RewriteOp(root)
    with InsertionPoint(rewrite.add_body()):
      ty3 = TypeOp()
      newOp = OperationOp(name="foo.op", types=[ty1, ty3])
      ReplaceOp(root, with_op=newOp)

# CHECK: module  {
# CHECK:   pdl.pattern @rewrite_type : benefit(1)  {
# CHECK:     %0 = type : i32
# CHECK:     %1 = type
# CHECK:     %2 = operation  -> (%0, %1 : !pdl.type, !pdl.type)
# CHECK:     rewrite %2  {
# CHECK:       %3 = operation "foo.op"  -> (%0, %1 : !pdl.type, !pdl.type)
# CHECK:     }
# CHECK:   }
# CHECK: }
@constructAndPrintInModule
def test_rewrite_type():
  pattern = PatternOp(1, "rewrite_type")
  with InsertionPoint(pattern.body):
    ty1 = TypeOp(IntegerType.get_signless(32))
    ty2 = TypeOp()
    root = OperationOp(types=[ty1, ty2])
    rewrite = RewriteOp(root)
    with InsertionPoint(rewrite.add_body()):
      newOp = OperationOp(name="foo.op", types=[ty1, ty2])

# CHECK: module  {
# CHECK:   pdl.pattern @rewrite_types : benefit(1)  {
# CHECK:     %0 = types
# CHECK:     %1 = operation  -> (%0 : !pdl.range<type>)
# CHECK:     rewrite %1  {
# CHECK:       %2 = types : [i32, i64]
# CHECK:       %3 = operation "foo.op"  -> (%0, %2 : !pdl.range<type>, !pdl.range<type>)
# CHECK:     }
# CHECK:   }
# CHECK: }
@constructAndPrintInModule
def test_rewrite_types():
  pattern = PatternOp(1, "rewrite_types")
  with InsertionPoint(pattern.body):
    types = TypesOp()
    root = OperationOp(types=[types])
    rewrite = RewriteOp(root)
    with InsertionPoint(rewrite.add_body()):
      otherTypes = TypesOp([IntegerType.get_signless(32), IntegerType.get_signless(64)])
      newOp = OperationOp(name="foo.op", types=[types, otherTypes])

# CHECK: module  {
# CHECK:   pdl.pattern @rewrite_operands : benefit(1)  {
# CHECK:     %0 = types
# CHECK:     %1 = operands : %0
# CHECK:     %2 = operation(%1 : !pdl.range<value>)
# CHECK:     rewrite %2  {
# CHECK:       %3 = operation "foo.op"  -> (%0 : !pdl.range<type>)
# CHECK:     }
# CHECK:   }
# CHECK: }
@constructAndPrintInModule
def test_rewrite_operands():
  pattern = PatternOp(1, "rewrite_operands")
  with InsertionPoint(pattern.body):
    types = TypesOp()
    operands = OperandsOp(types)
    root = OperationOp(args=[operands])
    rewrite = RewriteOp(root)
    with InsertionPoint(rewrite.add_body()):
      newOp = OperationOp(name="foo.op", types=[types])

# CHECK: module  {
# CHECK:   pdl.pattern @native_rewrite : benefit(1)  {
# CHECK:     %0 = operation
# CHECK:     rewrite %0  {
# CHECK:       apply_native_rewrite "NativeRewrite"(%0 : !pdl.operation)
# CHECK:     }
# CHECK:   }
# CHECK: }
@constructAndPrintInModule
def test_native_rewrite():
  pattern = PatternOp(1, "native_rewrite")
  with InsertionPoint(pattern.body):
    root = OperationOp()
    rewrite = RewriteOp(root)
    with InsertionPoint(rewrite.add_body()):
      ApplyNativeRewriteOp([], "NativeRewrite", args=[root])

# CHECK: module  {
# CHECK:   pdl.pattern @attribute_with_value : benefit(1)  {
# CHECK:     %0 = operation
# CHECK:     rewrite %0  {
# CHECK:       %1 = attribute "value"
# CHECK:       apply_native_rewrite "NativeRewrite"(%1 : !pdl.attribute)
# CHECK:     }
# CHECK:   }
# CHECK: }
@constructAndPrintInModule
def test_attribute_with_value():
  pattern = PatternOp(1, "attribute_with_value")
  with InsertionPoint(pattern.body):
    root = OperationOp()
    rewrite = RewriteOp(root)
    with InsertionPoint(rewrite.add_body()):
      attr = AttributeOp(value=Attribute.parse('"value"'))
      ApplyNativeRewriteOp([], "NativeRewrite", args=[attr])

# CHECK: module  {
# CHECK:   pdl.pattern @erase : benefit(1)  {
# CHECK:     %0 = operation
# CHECK:     rewrite %0  {
# CHECK:       erase %0
# CHECK:     }
# CHECK:   }
# CHECK: }
@constructAndPrintInModule
def test_erase():
  pattern = PatternOp(1, "erase")
  with InsertionPoint(pattern.body):
    root = OperationOp()
    rewrite = RewriteOp(root)
    with InsertionPoint(rewrite.add_body()):
      EraseOp(root)

# CHECK: module  {
# CHECK:   pdl.pattern @operation_results : benefit(1)  {
# CHECK:     %0 = types
# CHECK:     %1 = operation  -> (%0 : !pdl.range<type>)
# CHECK:     %2 = results of %1
# CHECK:     %3 = operation(%2 : !pdl.range<value>)
# CHECK:     rewrite %3 with "rewriter"
# CHECK:   }
# CHECK: }
@constructAndPrintInModule
def test_operation_results():
  valueRange = RangeType.get(ValueType.get())
  pattern = PatternOp(1, "operation_results")
  with InsertionPoint(pattern.body):
    types = TypesOp()
    inputOp = OperationOp(types=[types])
    results = ResultsOp(valueRange, inputOp)
    root = OperationOp(args=[results])
    RewriteOp(root, name="rewriter")

# CHECK: module  {
# CHECK:   pdl.pattern : benefit(1)  {
# CHECK:     %0 = type
# CHECK:     apply_native_constraint "typeConstraint"(%0 : !pdl.type)
# CHECK:     %1 = operation  -> (%0 : !pdl.type)
# CHECK:     rewrite %1 with "rewrite"
# CHECK:   }
# CHECK: }
@constructAndPrintInModule
def test_apply_native_constraint():
  pattern = PatternOp(1)
  with InsertionPoint(pattern.body):
    resultType = TypeOp()
    ApplyNativeConstraintOp("typeConstraint", args=[resultType])
    root = OperationOp(types=[resultType])
    RewriteOp(root, name="rewrite")
