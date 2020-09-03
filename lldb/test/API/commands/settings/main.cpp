#include <cstdlib>
#include <string>
#include <fstream>
#include <iostream>

int numberfn()
{
    return 0x5a;
}

int
main(int argc, char const *argv[])
{
    // The program writes its output to the following file:
    //
    //     o "output1.txt" for test_pass_host_env_vars() test case
    //     o "output2.txt" for test_run_args_and_env_vars_with_dsym() test case
    //     o "output2.txt" for test_run_args_and_env_vars_with_dwarf() test case
    std::ofstream outfile;
    if (argc == 1)
        outfile.open("output1.txt");
    else
        outfile.open("output2.txt");

    for (unsigned i = 0; i < argc; ++i) {
        std::string theArg(argv[i]);
        if (i == 1 && "A" == theArg)
            outfile << "argv[1] matches\n";

        if (i == 2 && "B" == theArg)
            outfile << "argv[2] matches\n";

        if (i == 3 && "C" == theArg)
            outfile << "argv[3] matches\n";
    }

    // For passing environment vars from the debugger to the launched process.
    if (::getenv("MY_ENV_VAR")) {
        std::string MY_ENV_VAR(getenv("MY_ENV_VAR"));
        if ("YES" == MY_ENV_VAR) {
            outfile << "Environment variable 'MY_ENV_VAR' successfully passed.\n";
        }
    }


    // For passing host environment vars to the launched process.
    if (::getenv("MY_HOST_ENV_VAR1")) {
        std::string MY_HOST_ENV_VAR1(getenv("MY_HOST_ENV_VAR1"));
        if ("VAR1" == MY_HOST_ENV_VAR1) {
            outfile << "The host environment variable 'MY_HOST_ENV_VAR1' successfully passed.\n";
        }
    }

    if (::getenv("MY_HOST_ENV_VAR2")) {
        std::string MY_HOST_ENV_VAR2(getenv("MY_HOST_ENV_VAR2"));
        if ("VAR2" == MY_HOST_ENV_VAR2) {
            outfile << "The host environment variable 'MY_HOST_ENV_VAR2' successfully passed.\n";
        }
    }

    std::cerr << "This message should go to standard error.\n";
    std::cout << "This message should go to standard out.\n";

    outfile.close();
    return numberfn();
}
