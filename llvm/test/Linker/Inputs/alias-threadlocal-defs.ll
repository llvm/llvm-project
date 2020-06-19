@tlsvar1 = thread_local global i32 0, align 4
@tlsvar2 = hidden thread_local alias i32, i32* @tlsvar1
