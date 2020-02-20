#include <os/activity.h>
#include <os/log.h>
#include <stdio.h>

#include "../common/darwin_log_common.h"

int main(int argc, char** argv)
{
    os_log_t logger_sub1 = os_log_create("org.llvm.lldb.test.sub1", "cat1");
    if (!logger_sub1)
        return 1;

    // Note we cannot use the os_log() line as the breakpoint because, as of
    // the initial writing of this test, we get multiple breakpoints for that
    // line, which confuses the pexpect test logic.
    printf("About to log\n"); // break here
    os_activity_t parent_activity = os_activity_create("parent-activity",
                                                       OS_ACTIVITY_CURRENT, OS_ACTIVITY_FLAG_DEFAULT);
    os_activity_apply(parent_activity, ^{
        os_activity_t child_activity = os_activity_create("child-activity",
                                                          OS_ACTIVITY_CURRENT, OS_ACTIVITY_FLAG_DEFAULT);
        os_activity_apply(child_activity, ^{
            os_log(logger_sub1, "This is the log message.");
        });
    });

    // Sleep, as the darwin log reporting doesn't always happen until a bit
    // later.  We need the message to come out before the process terminates.
    sleep(FINAL_WAIT_SECONDS);

    return 0;
}
