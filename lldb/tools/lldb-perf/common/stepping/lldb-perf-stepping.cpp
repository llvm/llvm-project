#include <CoreFoundation/CoreFoundation.h>

#include "lldb-perf/lib/Timer.h"
#include "lldb-perf/lib/Metric.h"
#include "lldb-perf/lib/Measurement.h"
#include "lldb-perf/lib/TestCase.h"
#include "lldb-perf/lib/Xcode.h"

#include <unistd.h>
#include <string>
#include <getopt.h>

using namespace lldb_perf;

class StepTest : public TestCase
{
    typedef void (*no_function) (void);
    
public:
    StepTest(bool use_single_stepping = false) :
        m_main_source("stepping-testcase.cpp"),
        m_use_single_stepping(use_single_stepping),
        m_time_measurements(nullptr)
    {
    }
    
    virtual
    ~StepTest() {}
    
    virtual bool
    Setup (int& argc, const char**& argv)
    {
        TestCase::Setup (argc, argv);
        
        // Toggle the fast stepping command on or off as required.
        const char *single_step_cmd = "settings set target.use-fast-stepping false";
        const char *fast_step_cmd   = "settings set target.use-fast-stepping true";
        const char *cmd_to_use;
        
        if (m_use_single_stepping)
            cmd_to_use = single_step_cmd;
        else
            cmd_to_use = fast_step_cmd;
        
        SBCommandReturnObject return_object;
        m_debugger.GetCommandInterpreter().HandleCommand(cmd_to_use,
                                                         return_object);
        if (!return_object.Succeeded())
        {
            if (return_object.GetError() != NULL)
                printf ("Got an error running settings set: %s.\n", return_object.GetError());
            else
                printf ("Failed running settings set, no error.\n");
        }

        m_target = m_debugger.CreateTarget(m_app_path.c_str());
        m_first_bp = m_target.BreakpointCreateBySourceRegex("Here is some code to stop at originally.", m_main_source);
        
        const char* file_arg = m_app_path.c_str();
        const char* empty = nullptr;
        const char* args[] = {file_arg, empty};
        SBLaunchInfo launch_info (args);
        
        return Launch (launch_info);
    }

    void
    WriteResults (Results &results)
    {
        // Gotta turn off the last timer now.
        m_individual_step_times.push_back(m_time_measurements.Stop());

        size_t num_time_measurements = m_individual_step_times.size();
        
        Results::Dictionary& results_dict = results.GetDictionary();
        const char *short_format_string = "step-time-%0.2d";
        const size_t short_size = strlen(short_format_string) + 5;
        char short_buffer[short_size];
        const char *long_format_string  = "The time it takes for step %d in the step sequence.";
        const size_t long_size = strlen(long_format_string) + 5;
        char long_buffer[long_size];
        
        for (size_t i = 0; i < num_time_measurements; i++)
        {
            snprintf (short_buffer, short_size, short_format_string, i);
            snprintf (long_buffer, long_size, long_format_string, i);
            
            results_dict.AddDouble(short_buffer,
                                   long_buffer,
                                   m_individual_step_times[i]);

        }
        results_dict.AddDouble ("total-time", "Total time spent stepping.", m_time_measurements.GetMetric().GetSum());
        results_dict.AddDouble ("stddev-time", "StdDev of time spent stepping.", m_time_measurements.GetMetric().GetStandardDeviation());

        results.Write(m_out_path.c_str());
    }
    

    const char *
    GetExecutablePath () const
    {
        if (m_app_path.empty())
            return NULL;
        return m_app_path.c_str();
    }

    const char *
    GetResultFilePath () const
    {
        if (m_out_path.empty())
            return NULL;
        return m_out_path.c_str();
    }

    void
    SetExecutablePath (const char *path)
    {
        if (path && path[0])
            m_app_path = path;
        else
            m_app_path.clear();
    }
    
    void
    SetResultFilePath (const char *path)
    {
        if (path && path[0])
            m_out_path = path;
        else
            m_out_path.clear();
    }
    
    void
    SetUseSingleStep (bool use_it)
    {
        m_use_single_stepping = use_it;
    }
private:
    virtual void
	TestStep (int counter, ActionWanted &next_action)
    {
        if (counter > 0)
        {
            m_individual_step_times.push_back(m_time_measurements.Stop());
            
        }

        // Disable the breakpoint, just in case it gets multiple locations we don't want that confusing the stepping.
        if (counter == 0)
            m_first_bp.SetEnabled(false);

        next_action.StepOver(m_process.GetThreadAtIndex(0));
        m_time_measurements.Start();

    
    }
    
    SBBreakpoint m_first_bp;
    SBFileSpec   m_main_source;
    TimeMeasurement<no_function> m_time_measurements;
    std::vector<double>          m_individual_step_times;
    bool m_use_single_stepping;
    std::string m_app_path;
    std::string m_out_path;
    

};

struct Options
{
    std::string test_file_path;
    std::string out_file;
    bool verbose;
    bool fast_step;
    bool error;
    bool print_help;
    
    Options() :
        verbose (false),
        fast_step (true),
        error (false),
        print_help (false)
    {
    }
};

static struct option g_long_options[] = {
    { "verbose",      no_argument,            NULL, 'v' },
    { "single-step",  no_argument,            NULL, 's' },
    { "test-file",    required_argument,      NULL, 't' },
    { "out-file",     required_argument,      NULL, 'o' },
    { NULL,           0,                      NULL,  0  }
};


std::string
GetShortOptionString (struct option *long_options)
{
    std::string option_string;
    for (int i = 0; long_options[i].name != NULL; ++i)
    {
        if (long_options[i].flag == NULL)
        {
            option_string.push_back ((char) long_options[i].val);
            switch (long_options[i].has_arg)
            {
                default:
                case no_argument:
                    break;
                case required_argument:
                    option_string.push_back (':');
                    break;
                case optional_argument:
                    option_string.append (2, ':');
                    break;
            }
        }
    }
    return option_string;
}

int main(int argc, const char * argv[])
{

    // Prepare for & make calls to getopt_long_only.
    
    std::string short_option_string (GetShortOptionString(g_long_options));
    
    StepTest test;

    Options option_data;
    bool done = false;

#if __GLIBC__
    optind = 0;
#else
    optreset = 1;
    optind = 1;
#endif
    while (!done)
    {
        int long_options_index = -1;
        const int short_option = ::getopt_long_only (argc,
                                                     const_cast<char **>(argv),
                                                     short_option_string.c_str(),
                                                     g_long_options,
                                                     &long_options_index);
        
        switch (short_option)
        {
            case 0:
                // Already handled
                break;

            case -1:
                done = true;
                break;

            case '?':
                option_data.print_help = true;
                break;

            case 'h':
                option_data.print_help = true;
                break;
                
            case 'v':
                option_data.verbose = true;
                break;
                
            case 's':
                option_data.fast_step = false;
                test.SetUseSingleStep(true);
                break;
                
            case 't':
                {
                    SBFileSpec file(optarg);
                    if (file.Exists())
                        test.SetExecutablePath(optarg);
                    else
                        fprintf(stderr, "error: file specified in --test-file (-t) option doesn't exist: '%s'\n", optarg);
                }
                break;
                
            case 'o':
                test.SetResultFilePath(optarg);
                break;
                
            default:
                option_data.error = true;
                option_data.print_help = true;
                fprintf (stderr, "error: unrecognized option %c\n", short_option);
                break;
        }
    }


    if (option_data.print_help)
    {
        puts(R"(
NAME
    lldb-perf-stepping -- a tool that measures LLDB peformance of simple stepping operations.

SYNOPSIS
    lldb-perf-stepping --test-file=FILE [--out-file=PATH --verbose --fast-step]
             
DESCRIPTION
    Runs a set of stepping operations, timing each step and outputs results
    to a plist file.
)");
        exit(0);
    }
    if (option_data.error)
    {
        exit(1);
    }

    if (test.GetExecutablePath() == NULL)
    {
        // --clang is mandatory
        option_data.print_help = true;
        option_data.error = true;
        fprintf (stderr, "error: the '--test-file=PATH' option is mandatory\n");
    }

    // Update argc and argv after parsing options
    argc -= optind;
    argv += optind;

    test.SetVerbose(true);
    TestCase::Run(test, argc, argv);
    return 0;
}
