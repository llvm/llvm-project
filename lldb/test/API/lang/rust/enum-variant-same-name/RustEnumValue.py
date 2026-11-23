"""Helper library to traverse data emitted for Rust enums """
from lldbsuite.test.lldbtest import *

DISCRIMINANT_MEMBER_NAME = "$discr$"
VALUE_MEMBER_NAME = "value"


class RustEnumValue:
    def __init__(self, value: lldb.SBValue):
        self.value = value

    def getAllVariantTypes(self):
        result = []
        for i in range(self._inner().GetNumChildren()):
            result.append(self.getVariantByIndex(i).GetDisplayTypeName())
        return result

    def _inner(self) -> lldb.SBValue:
        return self.value.GetChildAtIndex(0)

    def getVariantByIndex(self, index):
        return (
            self._inner()
            .GetChildAtIndex(index)
            .GetChildMemberWithName(VALUE_MEMBER_NAME)
        )

    @staticmethod
    def _getDiscriminantValueAsUnsigned(discr_sbvalue: lldb.SBValue):
        byte_size = discr_sbvalue.GetType().GetByteSize()
        error = lldb.SBError()

        # when discriminant is u16 Clang emits 'unsigned char'
        # and LLDB seems to treat it as character type disalowing to call GetValueAsUnsigned
        if byte_size == 1:
            return discr_sbvalue.GetData().GetUnsignedInt8(error, 0)
        elif byte_size == 2:
            return discr_sbvalue.GetData().GetUnsignedInt16(error, 0)
        elif byte_size == 4:
            return discr_sbvalue.GetData().GetUnsignedInt32(error, 0)
        elif byte_size == 8:
            return discr_sbvalue.GetData().GetUnsignedInt64(error, 0)
        else:
            return discr_sbvalue.GetValueAsUnsigned()

    def getCurrentVariantIndex(self):
        default_index = 0
        for i in range(self._inner().GetNumChildren()):
            variant: lldb.SBValue = self._inner().GetChildAtIndex(i)
            discr = variant.GetChildMemberWithName(DISCRIMINANT_MEMBER_NAME)
            if discr.IsValid():
                discr_unsigned_value = RustEnumValue._getDiscriminantValueAsUnsigned(
                    discr
                )
                if variant.GetName() == f"$variant${discr_unsigned_value}":
                    return discr_unsigned_value
            else:
                default_index = i
        return default_index

    def getFields(self):
        result = []
        for i in range(self._inner().GetNumChildren()):
            type: lldb.SBType = self._inner().GetType()
            result.append(type.GetFieldAtIndex(i).GetName())
        return result

    def getCurrentValue(self) -> lldb.SBValue:
        return self.getVariantByIndex(self.getCurrentVariantIndex())
