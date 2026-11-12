#include <dlfcn.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

int setup_is_complete = 0;

int main(int argc, const char** argv)
{

    void *handle = dlopen ("com.apple.sbd.xpc/com.apple.sbd", RTLD_NOW);
    if (handle)
    {
        if (dlsym(handle, "foo"))
        {
            system ("/bin/rm -rf com.apple.sbd.xpc com.apple.sbd.xpc.dSYM");

            FILE *fp = fopen (argv[1], "w");
            fclose (fp);
            setup_is_complete = 1;

            // At this point we want lldb to attach to the process.  If lldb attaches
            // before we've removed the dlopen'ed bundle, lldb will find the bundle
            // at its actual filepath and not have to do any tricky work, invalidating
            // the test.

            for (int loop_limiter = 0; loop_limiter < 100; loop_limiter++)
                sleep (1);
        }
    }
    return 0;
}
