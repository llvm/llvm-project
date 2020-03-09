import lldb

def foo_SummaryProvider(valobj, dict):
    a = valobj.GetChildMemberWithName('a')
    a_ptr = valobj.GetChildMemberWithName('a_ptr')
    bar = valobj.GetChildMemberWithName('b')
    i = bar.GetChildMemberWithName('i')
    i_ptr = bar.GetChildMemberWithName('i_ptr')
    b_ref = bar.GetChildMemberWithName('b_ref')
    b_ref_ptr = b_ref.AddressOf()
    b_ref = b_ref_ptr.Dereference()
    h = b_ref.GetChildMemberWithName('h')
    k = b_ref.GetChildMemberWithName('k')
    return 'a = ' + str(a.GetValueAsUnsigned(0)) + ', a_ptr = ' + \
        str(a_ptr.GetValueAsUnsigned(0)) + ' -> ' + str(a_ptr.Dereference().GetValueAsUnsigned(0)) + \
        ', i = ' + str(i.GetValueAsUnsigned(0)) + \
        ', i_ptr = ' + str(i_ptr.GetValueAsUnsigned(0)) + ' -> ' + str(i_ptr.Dereference().GetValueAsUnsigned(0)) + \
        ', b_ref = ' + str(b_ref.GetValueAsUnsigned(0)) + \
        ', h = ' + str(h.GetValueAsUnsigned(0)) + ' , k = ' + str(k.GetValueAsUnsigned(0))

def foo_SummaryProvider3(valobj, dict, options):
    if not isinstance(options, lldb.SBTypeSummaryOptions):
        raise Exception()
    return foo_SummaryProvider(valobj, dict) + ", WITH_OPTS"