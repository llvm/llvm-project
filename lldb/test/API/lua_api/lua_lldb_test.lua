-- Make lldb available in global
lldb = require('lldb')

-- Global assertion functions
function assertTrue(x)
    if not x then error('assertTrue failure') end
end

function assertFalse(x)
    if x then error('assertNotNil failure') end
end

function assertNotNil(x)
    if x == nil then error('assertNotNil failure') end
end

function assertEqual(x, y)
    if type(x) == 'table' and type(y) == 'table' then
        for k, _ in pairs(x) do
            assertEqual(x[k], y[k])
        end
    elseif type(x) ~= type(y) then
        error('assertEqual failure')
    elseif x ~= y then
        error('assertEqual failure')
    end
end

function assertStrContains(x, y)
    if not string.find(x, y, 1, true) then
        error('assertStrContains failure')
    end
end

-- Global helper functions
function read_file_non_empty_lines(f)
    local lines = {}
    while true do
        local line = f:read('*l')
        if not line then break end
        if line ~= '\n' then table.insert(lines, line) end
    end
    return lines
end

function split_lines(str)
    local lines = {}
    for line in str:gmatch("[^\r\n]+") do
        table.insert(lines, line)
    end
    return lines
end

function get_stopped_threads(process, reason)
    local threads = {}
    for i = 0, process:GetNumThreads() - 1 do
        local t = process:GetThreadAtIndex(i)
        if t:IsValid() and t:GetStopReason() == reason then
            table.insert(threads, t)
        end
    end
    return threads
end

function get_stopped_thread(process, reason)
    local threads = get_stopped_threads(process, reason)
    if #threads ~= 0 then return threads[1]
    else return nil end
end

-- Test helper

local _M = {}
local _m = {}

local _mt = { __index = _m }

function _M.create_test(name, exe, output, input)
    print('[lldb/lua] Create test ' .. name)
    exe = exe or os.getenv('TEST_EXE')
    output = output or os.getenv('TEST_OUTPUT')
    input = input or os.getenv('TEST_INPUT')
    lldb.SBDebugger.Initialize()
    local debugger = lldb.SBDebugger.Create()
    -- Ensure that debugger is created
    assertNotNil(debugger)
    assertTrue(debugger:IsValid())

    debugger:SetAsync(false)

    local lua_language = debugger:GetScriptingLanguage('lua')
    assertNotNil(lua_language)
    debugger:SetScriptLanguage(lua_language)

    local test = setmetatable({
        output = output,
        input = input,
        name = name,
        exe = exe,
        debugger = debugger
    }, _mt)
    _G[name] = test
    return test
end

function _m:create_target(exe)
    local target
    if not exe then exe = self.exe end
    target = self.debugger:CreateTarget(exe)
    -- Ensure that target is created
    assertNotNil(target)
    assertTrue(target:IsValid())
    return target
end

function _m:handle_command(command, collect)
    if collect == nil then collect = true end
    if collect then
        local ret = lldb.SBCommandReturnObject()
        local interpreter = self.debugger:GetCommandInterpreter()
        assertTrue(interpreter:IsValid())
        interpreter:HandleCommand(command, ret)
        self.debugger:GetOutputFile():Flush()
        self.debugger:GetErrorFile():Flush()
        assertTrue(ret:Succeeded())
        return ret:GetOutput()
    else
        self.debugger:HandleCommand(command)
        self.debugger:GetOutputFile():Flush()
        self.debugger:GetErrorFile():Flush()
    end
end

function _m:run()
    local tests = {}
    for k, v in pairs(self) do
        if string.sub(k, 1, 4) == 'Test' then
            table.insert(tests, k)
        end
    end
    table.sort(tests)
    for _, t in ipairs(tests) do
        print('[lldb/lua] Doing test ' .. self.name .. ' - ' .. t)
        local success = xpcall(self[t], function(e)
            print(debug.traceback())
        end, self)
        if not success then
            print('[lldb/lua] Failure in test ' .. self.name .. ' - ' .. t)
            return 1
        end
    end
    return 0
end

return _M
