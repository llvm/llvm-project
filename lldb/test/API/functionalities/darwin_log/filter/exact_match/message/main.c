#include <os/activity.h>
#include <os/log.h>
#include <stdio.h>

#include "../../../common/darwin_log_common.h"

int main(int argc, char** argv)
{
    os_log_t logger_sub1 = os_log_create("org.llvm.lldb.test.sub1", "cat1");
    os_log_t logger_sub2 = os_log_create("org.llvm.lldb.test.sub2", "cat2");
    if (!logger_sub1 || !logger_sub2)
        return 1;

    // Note we cannot use the os_log() line as the breakpoint because, as of
    // the initial writing of this test, we get multiple breakpoints for that
    // line, which confuses the pexpect test logic.
    printf("About to log\n"); // break here
    os_log(logger_sub1, "log message sub%d-cat%d", 1, 1);
    os_log(logger_sub2, "log message sub%d-cat%d", 2, 2);

    // Sleep, as the darwin log reporting doesn't always happen until a bit
    // later.  We need the message to come out before the process terminates.
    sleep(1);

    return 0;
}
