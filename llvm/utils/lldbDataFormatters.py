"""
LLDB Formatters for LLVM data types.

Load into LLDB with 'command script import /path/to/lldbDataFormatters.py'
"""

import lldb
import json

def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand('type category define -e llvm -l c++')
    debugger.HandleCommand('type synthetic add -w llvm '
                           '-l lldbDataFormatters.SmallVectorSynthProvider '
                           '-x "^llvm::SmallVectorImpl<.+>$"')
    debugger.HandleCommand('type summary add -w llvm '
                           '-s "size=${svar%#}" '
                           '-x "^llvm::SmallVectorImpl<.+>$"')
    debugger.HandleCommand('type synthetic add -w llvm '
                           '-l lldbDataFormatters.SmallVectorSynthProvider '
                           '-x "^llvm::SmallVector<.+,.+>$"')
    debugger.HandleCommand('type summary add -w llvm '
                           '-s "size=${svar%#}" '
                           '-x "^llvm::SmallVector<.+,.+>$"')
    debugger.HandleCommand('type synthetic add -w llvm '
                           '-l lldbDataFormatters.ArrayRefSynthProvider '
                           '-x "^llvm::ArrayRef<.+>$"')
    debugger.HandleCommand('type summary add -w llvm '
                           '-s "size=${svar%#}" '
                           '-x "^llvm::ArrayRef<.+>$"')
    debugger.HandleCommand('type synthetic add -w llvm '
                           '-l lldbDataFormatters.OptionalSynthProvider '
                           '-x "^llvm::Optional<.+>$"')
    debugger.HandleCommand('type summary add -w llvm '
                           '-F lldbDataFormatters.OptionalSummaryProvider '
                           '-x "^llvm::Optional<.+>$"')
    debugger.HandleCommand('type summary add -w llvm '
                           '-F lldbDataFormatters.SmallStringSummaryProvider '
                           '-x "^llvm::SmallString<.+>$"')
    debugger.HandleCommand('type summary add -w llvm '
                           '-F lldbDataFormatters.StringRefSummaryProvider '
                           '-x "^llvm::StringRef$"')
    debugger.HandleCommand('type summary add -w llvm '
                           '-F lldbDataFormatters.ConstStringSummaryProvider '
                           '-x "^lldb_private::ConstString$"')
    debugger.HandleCommand('type synthetic add -w llvm '
                           '-l lldbDataFormatters.PointerIntPairSynthProvider '
                           '-x "^llvm::PointerIntPair<.+>$"')
    debugger.HandleCommand('type synthetic add -w llvm '
                           '-l lldbDataFormatters.PointerUnionSynthProvider '
                           '-x "^llvm::PointerUnion<.+>$"')


# Pretty printer for llvm::SmallVector/llvm::SmallVectorImpl
class SmallVectorSynthProvider:
    def __init__(self, valobj, internal_dict):
        self.valobj = valobj;
        self.update() # initialize this provider

    def num_children(self):
        return self.size.GetValueAsUnsigned(0)

    def get_child_index(self, name):
        try:
            return int(name.lstrip('[').rstrip(']'))
        except:
            return -1;

    def get_child_at_index(self, index):
        # Do bounds checking.
        if index < 0:
            return None
        if index >= self.num_children():
            return None;

        offset = index * self.type_size
        return self.begin.CreateChildAtOffset('['+str(index)+']',
                                              offset, self.data_type)

    def update(self):
        self.begin = self.valobj.GetChildMemberWithName('BeginX')
        self.size = self.valobj.GetChildMemberWithName('Size')
        the_type = self.valobj.GetType()
        # If this is a reference type we have to dereference it to get to the
        # template parameter.
        if the_type.IsReferenceType():
            the_type = the_type.GetDereferencedType()

        self.data_type = the_type.GetTemplateArgumentType(0)
        self.type_size = self.data_type.GetByteSize()
        assert self.type_size != 0

class ArrayRefSynthProvider:
    """ Provider for llvm::ArrayRef """
    def __init__(self, valobj, internal_dict):
        self.valobj = valobj;
        self.update() # initialize this provider

    def num_children(self):
        return self.length

    def get_child_index(self, name):
        try:
            return int(name.lstrip('[').rstrip(']'))
        except:
            return -1;

    def get_child_at_index(self, index):
        if index < 0 or index >= self.num_children():
            return None;
        offset = index * self.type_size
        return self.data.CreateChildAtOffset('[' + str(index) + ']',
                                             offset, self.data_type)

    def update(self):
        self.data = self.valobj.GetChildMemberWithName('Data')
        length_obj = self.valobj.GetChildMemberWithName('Length')
        self.length = length_obj.GetValueAsUnsigned(0)
        self.data_type = self.data.GetType().GetPointeeType()
        self.type_size = self.data_type.GetByteSize()
        assert self.type_size != 0

def GetOptionalValue(valobj):
    storage = valobj.GetChildMemberWithName('Storage')
    if not storage:
        storage = valobj

    failure = 2
    hasVal = storage.GetChildMemberWithName('hasVal').GetValueAsUnsigned(failure)
    if hasVal == failure:
        return '<could not read llvm::Optional>'

    if hasVal == 0:
        return None

    underlying_type = storage.GetType().GetTemplateArgumentType(0)
    storage = storage.GetChildMemberWithName('value')
    return storage.Cast(underlying_type)

def OptionalSummaryProvider(valobj, internal_dict):
    val = GetOptionalValue(valobj)
    return val.summary if val else 'None'

class OptionalSynthProvider:
    """Provides deref support to llvm::Optional<T>"""
    def __init__(self, valobj, internal_dict):
        self.valobj = valobj

    def num_children(self):
        return self.valobj.num_children

    def get_child_index(self, name):
        if name == '$$dereference$$':
            return self.valobj.num_children
        return self.valobj.GetIndexOfChildWithName(name)

    def get_child_at_index(self, index):
        if index < self.valobj.num_children:
            return self.valobj.GetChildAtIndex(index)
        return GetOptionalValue(self.valobj) or lldb.SBValue()

def SmallStringSummaryProvider(valobj, internal_dict):
    num_elements = valobj.GetNumChildren()
    res = "\""
    for i in range(0, num_elements):
        c = valobj.GetChildAtIndex(i).GetValue()
        if c:
            res += c.strip("'")
    res += "\""
    return res


def StringRefSummaryProvider(valobj, internal_dict):
    if valobj.GetNumChildren() == 2:
        # StringRef's are also used to point at binary blobs in memory,
        # so filter out suspiciously long strings.
        max_length = 1024
        actual_length = valobj.GetChildAtIndex(1).GetValueAsUnsigned()
        truncate = actual_length > max_length
        length = min(max_length, actual_length)
        if length == 0:
            return '""'

        data = valobj.GetChildAtIndex(0).GetPointeeData(item_count=length)
        error = lldb.SBError()
        string = data.ReadRawData(error, 0, data.GetByteSize()).decode()
        if error.Fail():
            return "<error: %s>" % error.description

        # json.dumps conveniently escapes the string for us.
        string = json.dumps(string)
        if truncate:
            string += "..."
        return string
    return None


def ConstStringSummaryProvider(valobj, internal_dict):
    if valobj.GetNumChildren() == 1:
        return valobj.GetChildAtIndex(0).GetSummary()
    return ""


def get_expression_path(val):
    stream = lldb.SBStream()
    if not val.GetExpressionPath(stream):
        return None
    return stream.GetData()


class PointerIntPairSynthProvider:
    def __init__(self, valobj, internal_dict):
        self.valobj = valobj
        self.update()

    def num_children(self):
        return 2

    def get_child_index(self, name):
        if name == 'Pointer':
            return 0
        if name == 'Int':
            return 1
        return None

    def get_child_at_index(self, index):
        expr_path = get_expression_path(self.valobj)
        if index == 0:
            return self.valobj.CreateValueFromExpression('Pointer', f'({self.pointer_ty.name}){expr_path}.getPointer()')
        if index == 1:
            return self.valobj.CreateValueFromExpression('Int', f'({self.int_ty.name}){expr_path}.getInt()')
        return None

    def update(self):
        self.pointer_ty = self.valobj.GetType().GetTemplateArgumentType(0)
        self.int_ty = self.valobj.GetType().GetTemplateArgumentType(2)


def parse_template_parameters(typename):
    """
    LLDB doesn't support template parameter packs, so let's parse them manually.
    """
    result = []
    start = typename.find('<')
    end = typename.rfind('>')
    if start < 1 or end < 2 or end - start < 2:
        return result

    nesting_level = 0
    current_parameter_start = start + 1

    for i in range(start + 1, end + 1):
        c = typename[i]
        if c == '<':
            nesting_level += 1
        elif c == '>':
            nesting_level -= 1
        elif c == ',' and nesting_level == 0:
            result.append(typename[current_parameter_start:i].strip())
            current_parameter_start = i + 1

    result.append(typename[current_parameter_start:i].strip())

    return result


class PointerUnionSynthProvider:
    def __init__(self, valobj, internal_dict):
        self.valobj = valobj
        self.update()

    def num_children(self):
        return 1

    def get_child_index(self, name):
        if name == 'Ptr':
            return 0
        return None

    def get_child_at_index(self, index):
        if index != 0:
            return None
        ptr_type_name = self.template_args[self.active_type_tag]
        return self.valobj.CreateValueFromExpression('Ptr', f'({ptr_type_name}){self.val_expr_path}.getPointer()')

    def update(self):
        self.pointer_int_pair = self.valobj.GetChildMemberWithName('Val')
        self.val_expr_path = get_expression_path(self.valobj.GetChildMemberWithName('Val'))
        self.active_type_tag = self.valobj.CreateValueFromExpression('', f'(int){self.val_expr_path}.getInt()').GetValueAsSigned()
        self.template_args = parse_template_parameters(self.valobj.GetType().name)
