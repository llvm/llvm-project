_T = require('lua_lldb_test').create_test('TestThreadAPI')

function _T:TestGetStopDescription()
    local target = self:create_target()
    local breakpoint = target:BreakpointCreateByName("main", "a.out")
    assertTrue(breakpoint:IsValid() and breakpoint:GetNumLocations() == 1)

    local process = target:LaunchSimple({ 'arg1', 'arg2' }, nil, nil)
    local thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
    assertNotNil(thread)
    assertTrue(thread:IsValid())

    assertEqual("breakpoint", thread:GetStopDescription(string.len("breakpoint") + 1))
    assertEqual("break", thread:GetStopDescription(string.len("break") + 1))
    assertEqual("b", thread:GetStopDescription(string.len("b") + 1))
    assertEqual("breakpoint 1.1", thread:GetStopDescription(string.len("breakpoint 1.1") + 100))

    -- Test stream variation
    local stream = lldb.SBStream()
    assertTrue(thread:GetStopDescription(stream))
    assertNotNil(stream)
    assertEqual("breakpoint 1.1", stream:GetData())
end

os.exit(_T:run())
