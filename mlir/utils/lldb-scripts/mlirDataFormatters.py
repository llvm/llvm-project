"""
LLDB Formatters for MLIR data types.

Load into LLDB with 'command script import /path/to/mlirDataFormatters.py'
"""

import re
import lldb


def get_expression_path(val: lldb.SBValue):
    """Compute the expression path for the given value."""

    stream = lldb.SBStream()
    if not val.GetExpressionPath(stream):
        return None
    return stream.GetData()


def build_ptr_str_from_addr(addrValue: lldb.SBValue, type: lldb.SBType):
    """Build a string that computes a pointer using the given address value and type."""

    if type.is_reference:
        type = type.GetDereferencedType()
    if not type.is_pointer:
        type = type.GetPointerType()
    return f"(({type}){addrValue.GetData().GetUnsignedInt64(lldb.SBError(), 0)})"


# ===----------------------------------------------------------------------=== #
# Attributes and Types
# ===----------------------------------------------------------------------=== #

# This variable defines various mnemonic strings for use by the builtin
# dialect attributes and types, which often have special formatting within
# the parser/printer.
builtin_attr_type_mnemonics = {
    "mlir::AffineMapAttr": '"affine_map<...>"',
    "mlir::ArrayAttr": '"[...]"',
    "mlir::DenseArray": '"array<...>"',
    "mlir::DenseResourceElementsAttr": '"dense_resource<...>"',
    "mlir::DictionaryAttr": '"{...}"',
    "mlir::IntegerAttr": '"float"',
    "mlir::IntegerAttr": '"integer"',
    "mlir::IntegerSetAttr": '"affine_set<...>"',
    "mlir::SparseElementsAttr": '"sparse<...>"',
    "mlir::StringAttr": '""...""',
    "mlir::StridedLayout": '"strided_layout"',
    "mlir::UnitAttr": '"unit"',
    "mlir::CallSiteLoc": '"loc(callsite(...))"',
    "mlir::FusedLoc": '"loc(fused<...>[...])"',
    "mlir::UnknownLoc": '"loc(unknown)"',
    "mlir::Float8E5M2Type": '"f8E5M2"',
    "mlir::Float8E4M3FNType": '"f8E4M3FN"',
    "mlir::Float8E5M2FNUZType": '"f8E5M2FNUZ"',
    "mlir::Float8E4M3FNUZType": '"f8E4M3FNUZ"',
    "mlir::BFloat16Type": '"bf16"',
    "mlir::Float16Type": '"f16"',
    "mlir::Float32Type": '"f32"',
    "mlir::Float64Type": '"f64"',
    "mlir::Float80Type": '"f80"',
    "mlir::Float128Type": '"f128"',
    "mlir::FunctionType": '"(...) -> (...)"',
    "mlir::IndexType": '"index"',
    "mlir::IntegerType": '"iN"',
    "mlir::NoneType": '"none"',
    "mlir::TupleType": '"tuple<...>"',
    "mlir::MemRefType": '"memref<...>"',
    "mlir::UnrankedMemRef": '"memref<...>"',
    "mlir::UnrankedTensorType": '"tensor<...>"',
    "mlir::RankedTensorType": '"tensor<...>"',
    "mlir::VectorType": '"vector<...>"',
}


class ComputedTypeIDMap:
    """Compute a map of type ids to derived attributes, types, and locations.

    This is necessary for determining the C++ type when holding a base class,
    where we really only have access to dynamic information.
    """

    def __init__(self, target: lldb.SBTarget, internal_dict: dict):
        self.resolved_typeids = {}

        # Find all of the `id` variables, which are the name of TypeID variables
        # defined within the TypeIDResolver.
        type_ids = target.FindGlobalVariables("id", lldb.UINT32_MAX)
        for type_id in type_ids:
            # Strip out any matches that didn't come from a TypeID resolver. This
            # also lets us extract the derived type name.
            name = type_id.GetName()
            match = re.search("^mlir::detail::TypeIDResolver<(.*), void>::id$", name)
            if not match:
                continue
            type_name = match.group(1)

            # Filter out types that we don't care about.
            if not type_name.endswith(("Attr", "Loc", "Type")):
                continue

            # Find the LLDB type for the derived type.
            type = None
            for typeIt in target.FindTypes(type_name):
                if not typeIt or not typeIt.IsValid():
                    continue
                type = typeIt
                break
            if not type or not type.IsValid():
                continue

            # Map the raw address of the type id variable to the LLDB type.
            self.resolved_typeids[type_id.AddressOf().GetValueAsUnsigned()] = type

    # Resolve the type for the given TypeID address.
    def resolve_type(self, typeIdAddr: lldb.SBValue):
        try:
            return self.resolved_typeids[typeIdAddr.GetValueAsUnsigned()]
        except KeyError:
            return None


def is_derived_attribute_or_type(sbtype: lldb.SBType, internal_dict):
    """Return if the given type is a derived attribute or type."""

    # We only expect an AttrBase/TypeBase base class.
    if sbtype.num_bases != 1:
        return False
    base_name = sbtype.GetDirectBaseClassAtIndex(0).GetName()
    return base_name.startswith(("mlir::Attribute::AttrBase", "mlir::Type::TypeBase"))


def get_typeid_map(target: lldb.SBTarget, internal_dict: dict):
    """Get or construct a TypeID map for the given target."""

    if "typeIdMap" not in internal_dict:
        internal_dict["typeIdMap"] = ComputedTypeIDMap(target, internal_dict)
    return internal_dict["typeIdMap"]


def is_attribute_or_type(sbtype: lldb.SBType, internal_dict):
    """Return if the given type is an attribute or type."""

    num_bases = sbtype.GetNumberOfDirectBaseClasses()
    typeName = sbtype.GetName()

    # We bottom out at Attribute/Type/Location.
    if num_bases == 0:
        return typeName in ["mlir::Attribute", "mlir::Type", "mlir::Location"]

    # Check the easy cases of AttrBase/TypeBase.
    if typeName.startswith(("mlir::Attribute::AttrBase", "mlir::Type::TypeBase")):
        return True

    # Otherwise, recurse into the base class.
    return is_attribute_or_type(
        sbtype.GetDirectBaseClassAtIndex(0).GetType(), internal_dict
    )


def resolve_attr_type_from_value(
    valobj: lldb.SBValue, abstractVal: lldb.SBValue, internal_dict
):
    """Resolve the derived C++ type of an Attribute/Type value."""

    # Derived attribute/types already have the desired type.
    if is_derived_attribute_or_type(valobj.GetType(), internal_dict):
        return valobj.GetType()

    # Otherwise, we need to resolve the ImplTy from the TypeID. This is
    # done dynamically, because we don't use C++ RTTI of any kind.
    typeIdMap = get_typeid_map(valobj.GetTarget(), internal_dict)
    return typeIdMap.resolve_type(
        abstractVal.GetChildMemberWithName("typeID").GetChildMemberWithName("storage")
    )


class AttrTypeSynthProvider:
    """Define an LLDB synthetic children provider for Attributes and Types."""

    def __init__(self, valobj: lldb.SBValue, internal_dict):
        self.valobj = valobj

        # Grab the impl variable, which if this is a Location needs to be
        # resolved through the LocationAttr impl variable.
        impl: lldb.SBValue = self.valobj.GetChildMemberWithName("impl")
        if self.valobj.GetTypeName() == "mlir::Location":
            impl = impl.GetChildMemberWithName("impl")
        self.abstractVal = impl.GetChildMemberWithName("abstractType")
        if not self.abstractVal.IsValid():
            self.abstractVal = impl.GetChildMemberWithName("abstractAttribute")

        self.type = resolve_attr_type_from_value(
            valobj, self.abstractVal, internal_dict
        )
        if not self.type:
            return

        # Grab the ImplTy from the resolved type. This is the 3rd template
        # argument of the base class.
        self.impl_type = (
            self.type.GetDirectBaseClassAtIndex(0).GetType().GetTemplateArgumentType(2)
        )
        self.impl_pointer_ty = self.impl_type.GetPointerType()
        self.num_fields = self.impl_type.GetNumberOfFields()

        # Optionally add a mnemonic field.
        type_name = self.type.GetName()
        if type_name in builtin_attr_type_mnemonics:
            self.mnemonic = builtin_attr_type_mnemonics[type_name]
        elif type_name.startswith("mlir::Dense"):
            self.mnemonic = "dense<...>"
        else:
            self.mnemonic = self.valobj.CreateValueFromExpression(
                "mnemonic", f"(llvm::StringRef){type_name}::getMnemonic()"
            )
            if not self.mnemonic.summary:
                self.mnemonic = None
        if self.mnemonic:
            self.num_fields += 1

    def num_children(self):
        if not self.impl_type:
            return 0
        return self.num_fields

    def get_child_index(self, name):
        if not self.impl_type:
            return None
        if self.mnemonic and name == "[mnemonic]":
            return self.impl_type.GetNumberOfFields()
        for i in range(self.impl_type.GetNumberOfFields()):
            if self.impl_type.GetFieldAtIndex(i).GetName() == name:
                return i
        return None

    def get_child_at_index(self, index):
        if not self.impl_type or index >= self.num_fields:
            return None

        impl: lldb.SBValue = self.valobj.GetChildMemberWithName("impl")
        impl_ptr: lldb.SBValue = self.valobj.CreateValueFromData(
            build_ptr_str_from_addr(impl, self.impl_pointer_ty),
            impl.GetData(),
            self.impl_pointer_ty,
        )

        # Check for the mnemonic field.
        if index == self.impl_type.GetNumberOfFields():
            return self.valobj.CreateValueFromExpression(
                "[mnemonic]", self.get_mnemonic_string(impl_ptr)
            )

        # Otherwise, we expect the index to be a field.
        field: lldb.SBTypeMember = self.impl_type.GetFieldAtIndex(index)

        # Build the field access by resolving through the impl variable.
        return impl_ptr.GetChildMemberWithName(field.GetName())

    def get_mnemonic_string(self, impl_ptr: lldb.SBValue):
        if isinstance(self.mnemonic, str):
            return self.mnemonic

        # If we don't already have the mnemonic in string form, compute
        # it from the dialect name and the mnemonic.
        dialect_name = self.abstractVal.GetChildMemberWithName(
            "dialect"
        ).GetChildMemberWithName("name")
        self.mnemonic = f'{dialect_name.summary}"."{self.mnemonic.summary}'
        return self.mnemonic


def AttrTypeSummaryProvider(valobj: lldb.SBValue, internal_dict):
    """Define an LLDB summary provider for Attributes and Types."""

    # Check for a value field.
    value = valobj.GetChildMemberWithName("value")
    if value and value.summary:
        return value.summary

    # Otherwise, try the mnemoic.
    mnemonic: lldb.SBValue = valobj.GetChildMemberWithName("[mnemonic]")
    if not mnemonic.summary:
        return ""
    mnemonicStr = mnemonic.summary.strip('"')

    # Handle a few extremely common builtin attributes/types.
    ## IntegerType
    if mnemonicStr == "iN":
        signedness = valobj.GetChildMemberWithName("signedness").GetValueAsUnsigned()
        prefix = "i"
        if signedness == 1:
            prefix = "si"
        elif signedness == 2:
            prefix = "ui"
        return f"{prefix}{valobj.GetChildMemberWithName('width').GetValueAsUnsigned()}"
    ## IntegerAttr
    if mnemonicStr == "integer":
        value = valobj.GetChildMemberWithName("value")
        bitwidth = value.GetChildMemberWithName("BitWidth").GetValueAsUnsigned()
        if bitwidth <= 64:
            intVal = (
                value.GetChildMemberWithName("U")
                .GetChildMemberWithName("VAL")
                .GetValueAsUnsigned()
            )

            if bitwidth == 1:
                return "true" if intVal else "false"
            return f"{intVal} : i{bitwidth}"

    return mnemonicStr


# ===----------------------------------------------------------------------=== #
# mlir::Block
# ===----------------------------------------------------------------------=== #


class BlockSynthProvider:
    """Define an LLDB synthetic children provider for Blocks."""

    def __init__(self, valobj, internal_dict):
        self.valobj = valobj

    def num_children(self):
        return 3

    def get_child_index(self, name):
        if name == "parent":
            return 0
        if name == "operations":
            return 1
        if name == "arguments":
            return 2
        return None

    def get_child_at_index(self, index):
        if index >= 3:
            return None
        if index == 1:
            return self.valobj.GetChildMemberWithName("operations")
        if index == 2:
            return self.valobj.GetChildMemberWithName("arguments")

        expr_path = build_ptr_str_from_addr(self.valobj, self.valobj.GetType())
        return self.valobj.CreateValueFromExpression(
            "parent", f"{expr_path}->getParent()"
        )


# ===----------------------------------------------------------------------=== #
# mlir::Operation
# ===----------------------------------------------------------------------=== #


def is_op(sbtype: lldb.SBType, internal_dict):
    """Return if the given type is an operation."""

    # Bottom out at OpState/Op.
    typeName = sbtype.GetName()
    if sbtype.GetNumberOfDirectBaseClasses() == 0:
        return typeName == "mlir::OpState"
    if typeName == "mlir::Operation" or typeName.startswith("mlir::Op<"):
        return True

    # Otherwise, recurse into the base class.
    return is_op(sbtype.GetDirectBaseClassAtIndex(0).GetType(), internal_dict)


class OperationSynthProvider:
    """Define an LLDB synthetic children provider for Operations."""

    def __init__(self, valobj, internal_dict):
        self.valobj = valobj
        self.fields = []
        self.update()

    def num_children(self):
        return len(self.fields)

    def get_child_index(self, name):
        try:
            return self.fields.index(name)
        except ValueError:
            return None

    def get_child_at_index(self, index):
        if index >= len(self.fields):
            return None
        name = self.fields[index]
        if name == "name":
            return self.opobj.GetChildMemberWithName("name")
        if name == "parent":
            return self.opobj.GetChildMemberWithName("block").Clone("parent")
        if name == "location":
            return self.opobj.GetChildMemberWithName("location")
        if name == "attributes":
            return self.opobj.GetChildMemberWithName("attrs")

        expr_path = build_ptr_str_from_addr(self.opobj, self.opobj.GetType())
        if name == "operands":
            return self.opobj.CreateValueFromExpression(
                "operands", f"{expr_path}->debug_getOperands()"
            )
        if name == "results":
            return self.opobj.CreateValueFromExpression(
                "results", f"{expr_path}->debug_getResults()"
            )
        if name == "successors":
            return self.opobj.CreateValueFromExpression(
                "successors", f"{expr_path}->debug_getSuccessors()"
            )
        if name == "regions":
            return self.opobj.CreateValueFromExpression(
                "regions", f"{expr_path}->debug_getRegions()"
            )
        return None

    def update(self):
        # If this is a derived operation, we need to resolve through the
        # state field.
        self.opobj = self.valobj
        if "mlir::Operation" not in self.valobj.GetTypeName():
            self.opobj = self.valobj.GetChildMemberWithName("state")

        self.fields = ["parent", "name", "location", "attributes"]
        if (
            self.opobj.GetChildMemberWithName("hasOperandStorage").GetValueAsUnsigned(0)
            != 0
        ):
            self.fields.append("operands")
        if self.opobj.GetChildMemberWithName("numResults").GetValueAsUnsigned(0) != 0:
            self.fields.append("results")
        if self.opobj.GetChildMemberWithName("numSuccs").GetValueAsUnsigned(0) != 0:
            self.fields.append("successors")
        if self.opobj.GetChildMemberWithName("numRegions").GetValueAsUnsigned(0) != 0:
            self.fields.append("regions")


def OperationSummaryProvider(valobj: lldb.SBValue, internal_dict):
    """Define an LLDB summary provider for Operations."""

    name = valobj.GetChildMemberWithName("name")
    if name and name.summary:
        return name.summary
    return ""


# ===----------------------------------------------------------------------=== #
# Ranges
# ===----------------------------------------------------------------------=== #


class DirectRangeSynthProvider:
    """Define an LLDB synthetic children provider for direct ranges, i.e. those
    with a base pointer that points to the type of element we want to display.
    """

    def __init__(self, valobj, internal_dict):
        self.valobj = valobj
        self.update()

    def num_children(self):
        return self.length

    def get_child_index(self, name):
        try:
            return int(name.lstrip("[").rstrip("]"))
        except:
            return None

    def get_child_at_index(self, index):
        if index >= self.num_children():
            return None
        offset = index * self.type_size
        return self.data.CreateChildAtOffset(f"[{index}]", offset, self.data_type)

    def update(self):
        length_obj = self.valobj.GetChildMemberWithName("count")
        self.length = length_obj.GetValueAsUnsigned(0)

        self.data = self.valobj.GetChildMemberWithName("base")
        self.data_type = self.data.GetType().GetPointeeType()
        self.type_size = self.data_type.GetByteSize()
        assert self.type_size != 0


class InDirectRangeSynthProvider:
    """Define an LLDB synthetic children provider for ranges
    that transform the underlying base pointer, e.g. to convert
    it to a different type depending on various characteristics
    (e.g. mlir::ValueRange).
    """

    def __init__(self, valobj, internal_dict):
        self.valobj = valobj
        self.update()

    def num_children(self):
        return self.length

    def get_child_index(self, name):
        try:
            return int(name.lstrip("[").rstrip("]"))
        except:
            return None

    def get_child_at_index(self, index):
        if index >= self.num_children():
            return None
        expr_path = get_expression_path(self.valobj)
        return self.valobj.CreateValueFromExpression(
            f"[{index}]", f"{expr_path}[{index}]"
        )

    def update(self):
        length_obj = self.valobj.GetChildMemberWithName("count")
        self.length = length_obj.GetValueAsUnsigned(0)


class IPListRangeSynthProvider:
    """Define an LLDB synthetic children provider for an IPList.
    """

    def __init__(self, valobj, internal_dict):
        self.valobj = valobj
        self.update()

    def num_children(self):
        sentinel = self.valobj.GetChildMemberWithName("Sentinel")
        sentinel_addr = sentinel.AddressOf().GetValueAsUnsigned(0)

        # Iterate the next pointers looking for the sentinel.
        count = 0
        current = sentinel.GetChildMemberWithName("Next")
        while current.GetValueAsUnsigned(0) != sentinel_addr:
            current = current.GetChildMemberWithName("Next")
            count += 1

        return count

    def get_child_index(self, name):
        try:
            return int(name.lstrip("[").rstrip("]"))
        except:
            return None

    def get_child_at_index(self, index):
        if index >= self.num_children():
            return None

        # Start from the sentinel and grab the next pointer.
        value: lldb.SBValue = self.valobj.GetChildMemberWithName("Sentinel")
        it = 0
        while it <= index:
            value = value.GetChildMemberWithName("Next")
            it += 1

        return value.CreateValueFromExpression(
            f"[{index}]",
            f"(({self.value_type})({value.GetTypeName()}){value.GetValueAsUnsigned()})",
        )

    def update(self):
        self.value_type = (
            self.valobj.GetType().GetTemplateArgumentType(0).GetPointerType()
        )


# ===----------------------------------------------------------------------=== #
# mlir::Value
# ===----------------------------------------------------------------------=== #


class ValueSynthProvider:
    """Define an LLDB synthetic children provider for Values.
    """

    def __init__(self, valobj, internal_dict):
        self.valobj = valobj
        self.update()

    def num_children(self):
        # 7: BlockArgument:
        #  index, type, owner, firstUse, location
        if self.kind == 7:
            return 5

        # 0-6: OpResult:
        #  index, type, owner, firstUse
        return 4

    def get_child_index(self, name):
        if name == "index":
            return 0
        if name == "type":
            return 1
        if name == "owner":
            return 2
        if name == "firstUse":
            return 3
        if name == "location":
            return 4
        return None

    def get_child_at_index(self, index):
        if index >= self.num_children():
            return None

        # Check if the current value is already an Impl struct.
        if self.valobj.GetTypeName().endswith("Impl"):
            impl_ptr_str = build_ptr_str_from_addr(
                self.valobj.AddressOf(), self.valobj.GetType().GetPointerType()
            )
        else:
            impl = self.valobj.GetChildMemberWithName("impl")
            impl_ptr_str = build_ptr_str_from_addr(impl, impl.GetType())

        # Cast to the derived Impl type.
        if self.kind == 7:
            derived_impl_str = f"((mlir::detail::BlockArgumentImpl *){impl_ptr_str})"
        elif self.kind == 6:
            derived_impl_str = f"((mlir::detail::OutOfLineOpResult *){impl_ptr_str})"
        else:
            derived_impl_str = f"((mlir::detail::InlineOpResult *){impl_ptr_str})"

        # Handle the shared fields when possible.
        if index == 1:
            return self.valobj.CreateValueFromExpression(
                "type", f"{derived_impl_str}->debug_getType()"
            )
        if index == 3:
            return self.valobj.CreateValueFromExpression(
                "firstUse", f"{derived_impl_str}->firstUse"
            )

        # Handle Block argument children.
        if self.kind == 7:
            impl = self.valobj.CreateValueFromExpression("impl", derived_impl_str)
            if index == 0:
                return impl.GetChildMemberWithName("index")
            if index == 2:
                return impl.GetChildMemberWithName("owner")
            if index == 4:
                return impl.GetChildMemberWithName("loc")

        # Handle OpResult children.
        if index == 0:
            # Handle the out of line case.
            if self.kind == 6:
                return self.valobj.CreateValueFromExpression(
                    "index", f"{derived_impl_str}->outOfLineIndex + 6"
                )
            return self.valobj.CreateValueFromExpression("index", f"{self.kind}")
        if index == 2:
            return self.valobj.CreateValueFromExpression(
                "owner", f"{derived_impl_str}->getOwner()"
            )
        return None

    def update(self):
        # Check if the current value is already an Impl struct.
        if self.valobj.GetTypeName().endswith("Impl"):
            impl_ptr_str = build_ptr_str_from_addr(
                self.valobj, self.valobj.GetType().GetPointerType()
            )
        else:
            impl = self.valobj.GetChildMemberWithName("impl")
            impl_ptr_str = build_ptr_str_from_addr(impl, impl.GetType())

        # Compute the kind of value we are dealing with.
        self.kind = self.valobj.CreateValueFromExpression(
            "kind", f"{impl_ptr_str}->debug_getKind()"
        ).GetValueAsUnsigned()


def ValueSummaryProvider(valobj: lldb.SBValue, internal_dict):
    """Define an LLDB summary provider for Values.
    """

    index = valobj.GetChildMemberWithName("index").GetValueAsUnsigned()
    # Check if this is a block argument or not (block arguments have locations).
    if valobj.GetChildMemberWithName("location").IsValid():
        summary = f"Block Argument {index}"
    else:
        owner_name = (
            valobj.GetChildMemberWithName("owner")
            .GetChildMemberWithName("name")
            .summary
        )
        summary = f"{owner_name} Result {index}"

    # Grab the type to help form the summary.
    type = valobj.GetChildMemberWithName("type")
    if type.summary:
        summary += f": {type.summary}"

    return summary


# ===----------------------------------------------------------------------=== #
# Initialization
# ===----------------------------------------------------------------------=== #


def __lldb_init_module(debugger: lldb.SBDebugger, internal_dict):
    cat: lldb.SBTypeCategory = debugger.CreateCategory("mlir")
    cat.SetEnabled(True)

    # Attributes and Types
    cat.AddTypeSummary(
        lldb.SBTypeNameSpecifier(
            "mlirDataFormatters.is_attribute_or_type", lldb.eFormatterMatchCallback
        ),
        lldb.SBTypeSummary.CreateWithFunctionName(
            "mlirDataFormatters.AttrTypeSummaryProvider"
        ),
    )
    cat.AddTypeSynthetic(
        lldb.SBTypeNameSpecifier(
            "mlirDataFormatters.is_attribute_or_type", lldb.eFormatterMatchCallback
        ),
        lldb.SBTypeSynthetic.CreateWithClassName(
            "mlirDataFormatters.AttrTypeSynthProvider"
        ),
    )

    # Operation
    cat.AddTypeSynthetic(
        lldb.SBTypeNameSpecifier("mlir::Block", lldb.eFormatterMatchExact),
        lldb.SBTypeSynthetic.CreateWithClassName(
            "mlirDataFormatters.BlockSynthProvider"
        ),
    )

    # NamedAttribute
    cat.AddTypeSummary(
        lldb.SBTypeNameSpecifier("mlir::NamedAttribute", lldb.eFormatterMatchExact),
        lldb.SBTypeSummary.CreateWithSummaryString("${var.name%S} = ${var.value%S}"),
    )

    # OperationName
    cat.AddTypeSummary(
        lldb.SBTypeNameSpecifier("mlir::OperationName", lldb.eFormatterMatchExact),
        lldb.SBTypeSummary.CreateWithSummaryString("${var.impl->name%S}"),
    )

    # Operation
    cat.AddTypeSummary(
        lldb.SBTypeNameSpecifier(
            "mlirDataFormatters.is_op", lldb.eFormatterMatchCallback
        ),
        lldb.SBTypeSummary.CreateWithFunctionName(
            "mlirDataFormatters.OperationSummaryProvider"
        ),
    )
    cat.AddTypeSynthetic(
        lldb.SBTypeNameSpecifier(
            "mlirDataFormatters.is_op", lldb.eFormatterMatchCallback
        ),
        lldb.SBTypeSynthetic.CreateWithClassName(
            "mlirDataFormatters.OperationSynthProvider"
        ),
    )

    # Ranges
    def add_direct_range_summary_and_synth(name):
        cat.AddTypeSummary(
            lldb.SBTypeNameSpecifier(name, lldb.eFormatterMatchExact),
            lldb.SBTypeSummary.CreateWithSummaryString("size=${svar%#}"),
        )
        cat.AddTypeSynthetic(
            lldb.SBTypeNameSpecifier(name, lldb.eFormatterMatchExact),
            lldb.SBTypeSynthetic.CreateWithClassName(
                "mlirDataFormatters.DirectRangeSynthProvider"
            ),
        )

    def add_indirect_range_summary_and_synth(name):
        cat.AddTypeSummary(
            lldb.SBTypeNameSpecifier(name, lldb.eFormatterMatchExact),
            lldb.SBTypeSummary.CreateWithSummaryString("size=${svar%#}"),
        )
        cat.AddTypeSynthetic(
            lldb.SBTypeNameSpecifier(name, lldb.eFormatterMatchExact),
            lldb.SBTypeSynthetic.CreateWithClassName(
                "mlirDataFormatters.InDirectRangeSynthProvider"
            ),
        )

    def add_iplist_range_summary_and_synth(name):
        cat.AddTypeSummary(
            lldb.SBTypeNameSpecifier(name, lldb.eFormatterMatchExact),
            lldb.SBTypeSummary.CreateWithSummaryString("size=${svar%#}"),
        )
        cat.AddTypeSynthetic(
            lldb.SBTypeNameSpecifier(name, lldb.eFormatterMatchExact),
            lldb.SBTypeSynthetic.CreateWithClassName(
                "mlirDataFormatters.IPListRangeSynthProvider"
            ),
        )

    add_direct_range_summary_and_synth("mlir::Operation::operand_range")
    add_direct_range_summary_and_synth("mlir::OperandRange")
    add_direct_range_summary_and_synth("mlir::Operation::result_range")
    add_direct_range_summary_and_synth("mlir::ResultRange")
    add_direct_range_summary_and_synth("mlir::SuccessorRange")
    add_indirect_range_summary_and_synth("mlir::ValueRange")
    add_indirect_range_summary_and_synth("mlir::TypeRange")
    add_iplist_range_summary_and_synth("mlir::Block::OpListType")
    add_iplist_range_summary_and_synth("mlir::Region::BlockListType")

    # Values
    def add_value_summary_and_synth(name):
        cat.AddTypeSummary(
            lldb.SBTypeNameSpecifier(name, lldb.eFormatterMatchExact),
            lldb.SBTypeSummary.CreateWithFunctionName(
                "mlirDataFormatters.ValueSummaryProvider"
            ),
        )
        cat.AddTypeSynthetic(
            lldb.SBTypeNameSpecifier(name, lldb.eFormatterMatchExact),
            lldb.SBTypeSynthetic.CreateWithClassName(
                "mlirDataFormatters.ValueSynthProvider"
            ),
        )

    add_value_summary_and_synth("mlir::BlockArgument")
    add_value_summary_and_synth("mlir::Value")
    add_value_summary_and_synth("mlir::OpResult")
    add_value_summary_and_synth("mlir::detail::OpResultImpl")
