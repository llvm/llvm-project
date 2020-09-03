"""
LLDB Formatters for LLVM data types.

Load into LLDB with 'command script import /path/to/lldbDataFormatters.py'
"""

def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand('type category define -e llvm -l c++')
    debugger.HandleCommand('type synthetic add -w llvm '
                           '-l lldbDataFormatters.SmallVectorSynthProvider '
                           '-x "^llvm::SmallVectorImpl<.+>$"')
    debugger.HandleCommand('type synthetic add -w llvm '
                           '-l lldbDataFormatters.SmallVectorSynthProvider '
                           '-x "^llvm::SmallVector<.+,.+>$"')
    debugger.HandleCommand('type synthetic add -w llvm '
                           '-l lldbDataFormatters.ArrayRefSynthProvider '
                           '-x "^llvm::ArrayRef<.+>$"')
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

# Pretty printer for llvm::SmallVector/llvm::SmallVectorImpl
class SmallVectorSynthProvider:
    def __init__(self, valobj, dict):
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
    def __init__(self, valobj, dict):
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

def OptionalSummaryProvider(valobj, internal_dict):
    storage = valobj.GetChildMemberWithName('Storage')
    if not storage:
        storage = valobj

    failure = 2
    hasVal = storage.GetChildMemberWithName('hasVal').GetValueAsUnsigned(failure)
    if hasVal == failure:
        return '<could not read llvm::Optional>'

    if hasVal == 0:
        return 'None'

    underlying_type = storage.GetType().GetTemplateArgumentType(0)
    storage = storage.GetChildMemberWithName('storage')
    return str(storage.Cast(underlying_type))

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
        max_length = 256
        length = valobj.GetChildAtIndex(1).GetValueAsUnsigned(max_length)
        if length == 0:
            return "NULL"
        if length < max_length:
            return valobj.GetChildAtIndex(0).GetSummary()
    return ""

def ConstStringSummaryProvider(valobj, internal_dict):
    if valobj.GetNumChildren() == 1:
        return valobj.GetChildAtIndex(0).GetSummary()
    return ""
