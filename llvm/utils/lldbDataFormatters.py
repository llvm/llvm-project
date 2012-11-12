"""
Load into LLDB with:
script import lldbDataFormatters
type synthetic add -x "^llvm::SmallVectorImpl<.+>$" -l lldbDataFormatters.SmallVectorSynthProvider
type synthetic add -x "^llvm::SmallVector<.+,.+>$" -l lldbDataFormatters.SmallVectorSynthProvider
"""

# Pretty printer for llvm::SmallVector/llvm::SmallVectorImpl
class SmallVectorSynthProvider:
    def __init__(self, valobj, dict):
        self.valobj = valobj;
        self.update() # initialize this provider

    def num_children(self):
        begin = self.begin.GetValueAsUnsigned(0)
        end = self.end.GetValueAsUnsigned(0)
        return (end - begin)/self.type_size

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
        self.end = self.valobj.GetChildMemberWithName('EndX')
        the_type = self.valobj.GetType()
        # If this is a reference type we have to dereference it to get to the
        # template parameter.
        if the_type.IsReferenceType():
            the_type = the_type.GetDereferencedType()

        self.data_type = the_type.GetTemplateArgumentType(0)
        self.type_size = self.data_type.GetByteSize()
        assert self.type_size != 0
