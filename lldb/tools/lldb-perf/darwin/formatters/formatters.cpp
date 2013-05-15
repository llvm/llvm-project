//===-- formatters.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <CoreFoundation/CoreFoundation.h>

#include "lldb-perf/lib/Timer.h"
#include "lldb-perf/lib/Metric.h"
#include "lldb-perf/lib/Measurement.h"
#include "lldb-perf/lib/TestCase.h"
#include "lldb-perf/lib/Xcode.h"

#include <iostream>
#include <unistd.h>
#include <fstream>

using namespace lldb_perf;

class FormattersTest : public TestCase
{
public:
    FormattersTest () : TestCase()
    {
        m_dump_std_vector_measurement = CreateTimeMeasurement([] (SBValue value) -> void {
            lldb_perf::Xcode::FetchVariable (value,1,false);
        }, "std-vector", "time to dump an std::vector");
        m_dump_std_list_measurement = CreateTimeMeasurement([] (SBValue value) -> void {
            lldb_perf::Xcode::FetchVariable (value,1,false);
        }, "std-list", "time to dump an std::list");
        m_dump_std_map_measurement = CreateTimeMeasurement([] (SBValue value) -> void {
            lldb_perf::Xcode::FetchVariable (value,1,false);
        }, "std-map", "time to dump an std::map");
        
        // use this in manual mode
        m_dump_std_string_measurement = CreateTimeMeasurement([] () -> void {
        }, "std-string", "time to dump an std::string");
        
        m_dump_nsstring_measurement = CreateTimeMeasurement([] (SBValue value) -> void {
            lldb_perf::Xcode::FetchVariable (value,0,false);
        }, "ns-string", "time to dump an NSString");
        
        m_dump_nsarray_measurement = CreateTimeMeasurement([] (SBValue value) -> void {
            lldb_perf::Xcode::FetchVariable (value,1,false);
        }, "ns-array", "time to dump an NSArray");
        
        m_dump_nsdictionary_measurement = CreateTimeMeasurement([] (SBValue value) -> void {
            lldb_perf::Xcode::FetchVariable (value,1,false);
        }, "ns-dictionary", "time to dump an NSDictionary");
        
        m_dump_nsset_measurement = CreateTimeMeasurement([] (SBValue value) -> void {
            lldb_perf::Xcode::FetchVariable (value,1,false);
        }, "ns-set", "time to dump an NSSet");
        
        m_dump_nsbundle_measurement = CreateTimeMeasurement([] (SBValue value) -> void {
            lldb_perf::Xcode::FetchVariable (value,1,false);
        }, "ns-bundle", "time to dump an NSBundle");
        
        m_dump_nsdate_measurement = CreateTimeMeasurement([] (SBValue value) -> void {
            lldb_perf::Xcode::FetchVariable (value,0,false);
        }, "ns-date", "time to dump an NSDate");
    }

    virtual
    ~FormattersTest ()
    {
    }
    
    virtual bool
	Setup (int& argc, const char**& argv)
    {
        m_app_path.assign(argv[1]);
        m_out_path.assign(argv[2]);
        m_target = m_debugger.CreateTarget(m_app_path.c_str());
        m_target.BreakpointCreateByName("main");
        SBLaunchInfo launch_info(argv);
        return Launch (launch_info);
    }
    
    void
    DoTest ()
    {
        SBFrame frame_zero(m_thread.GetFrameAtIndex(0));
        
        m_dump_nsarray_measurement(frame_zero.FindVariable("nsarray", lldb::eDynamicCanRunTarget));
        m_dump_nsarray_measurement(frame_zero.FindVariable("nsmutablearray", lldb::eDynamicCanRunTarget));

        m_dump_nsdictionary_measurement(frame_zero.FindVariable("nsdictionary", lldb::eDynamicCanRunTarget));
        m_dump_nsdictionary_measurement(frame_zero.FindVariable("nsmutabledictionary", lldb::eDynamicCanRunTarget));
        
        m_dump_nsstring_measurement(frame_zero.FindVariable("str0", lldb::eDynamicCanRunTarget));
        m_dump_nsstring_measurement(frame_zero.FindVariable("str1", lldb::eDynamicCanRunTarget));
        m_dump_nsstring_measurement(frame_zero.FindVariable("str2", lldb::eDynamicCanRunTarget));
        m_dump_nsstring_measurement(frame_zero.FindVariable("str3", lldb::eDynamicCanRunTarget));
        m_dump_nsstring_measurement(frame_zero.FindVariable("str4", lldb::eDynamicCanRunTarget));
        
        m_dump_nsdate_measurement(frame_zero.FindVariable("me", lldb::eDynamicCanRunTarget));
        m_dump_nsdate_measurement(frame_zero.FindVariable("cutie", lldb::eDynamicCanRunTarget));
        m_dump_nsdate_measurement(frame_zero.FindVariable("mom", lldb::eDynamicCanRunTarget));
        m_dump_nsdate_measurement(frame_zero.FindVariable("dad", lldb::eDynamicCanRunTarget));
        m_dump_nsdate_measurement(frame_zero.FindVariable("today", lldb::eDynamicCanRunTarget));
        
        m_dump_nsbundle_measurement(frame_zero.FindVariable("bundles", lldb::eDynamicCanRunTarget));
        m_dump_nsbundle_measurement(frame_zero.FindVariable("frameworks", lldb::eDynamicCanRunTarget));
        
        m_dump_nsset_measurement(frame_zero.FindVariable("nsset", lldb::eDynamicCanRunTarget));
        m_dump_nsset_measurement(frame_zero.FindVariable("nsmutableset", lldb::eDynamicCanRunTarget));
        
        m_dump_std_vector_measurement(frame_zero.FindVariable("vector", lldb::eDynamicCanRunTarget));
        m_dump_std_list_measurement(frame_zero.FindVariable("list", lldb::eDynamicCanRunTarget));
        m_dump_std_map_measurement(frame_zero.FindVariable("map", lldb::eDynamicCanRunTarget));

        auto sstr0 = frame_zero.FindVariable("sstr0", lldb::eDynamicCanRunTarget);
        auto sstr1 = frame_zero.FindVariable("sstr1", lldb::eDynamicCanRunTarget);
        auto sstr2 = frame_zero.FindVariable("sstr2", lldb::eDynamicCanRunTarget);
        auto sstr3 = frame_zero.FindVariable("sstr3", lldb::eDynamicCanRunTarget);
        auto sstr4 = frame_zero.FindVariable("sstr4", lldb::eDynamicCanRunTarget);
        
        m_dump_std_string_measurement.Start();
        Xcode::FetchVariable(sstr0,0,false);
        m_dump_std_string_measurement.Stop();
        
        m_dump_std_string_measurement.Start();
        Xcode::FetchVariable(sstr1,0,false);
        m_dump_std_string_measurement.Stop();

        m_dump_std_string_measurement.Start();
        Xcode::FetchVariable(sstr2,0,false);
        m_dump_std_string_measurement.Stop();

        m_dump_std_string_measurement.Start();
        Xcode::FetchVariable(sstr3,0,false);
        m_dump_std_string_measurement.Stop();

        m_dump_std_string_measurement.Start();
        Xcode::FetchVariable(sstr4,0,false);
        m_dump_std_string_measurement.Stop();
        
    }
    
	virtual void
	TestStep (int counter, ActionWanted &next_action)
    {
        switch (counter)
        {
            case 0:
                m_target.BreakpointCreateByLocation("fmts_tester.mm", 78);
                next_action.Continue();
                break;
            case 1:
                DoTest ();
                next_action.Continue();
                break;
            case 2:
                DoTest ();
                next_action.Continue();
                break;
            case 3:
                DoTest ();
                next_action.Continue();
                break;
            case 4:
                DoTest ();
                next_action.Continue();
                break;
            case 5:
                DoTest ();
                next_action.Continue();
                break;
            case 6:
                DoTest ();
                next_action.Continue();
                break;
            case 7:
                DoTest ();
                next_action.Continue();
                break;
            case 8:
                DoTest ();
                next_action.Continue();
                break;
            case 9:
                DoTest ();
                next_action.Continue();
                break;
            case 10:
                DoTest ();
                next_action.Continue();
                break;
            default:
                next_action.Kill();
                break;
        }
    }
    
    virtual void
    WriteResults (Results &results)
    {
        m_dump_std_vector_measurement.WriteAverageAndStandardDeviation(results);
        m_dump_std_list_measurement.WriteAverageAndStandardDeviation(results);
        m_dump_std_map_measurement.WriteAverageAndStandardDeviation(results);
        m_dump_std_string_measurement.WriteAverageAndStandardDeviation(results);
        
        m_dump_nsstring_measurement.WriteAverageAndStandardDeviation(results);
        m_dump_nsarray_measurement.WriteAverageAndStandardDeviation(results);
        m_dump_nsdictionary_measurement.WriteAverageAndStandardDeviation(results);
        m_dump_nsset_measurement.WriteAverageAndStandardDeviation(results);
        m_dump_nsbundle_measurement.WriteAverageAndStandardDeviation(results);
        m_dump_nsdate_measurement.WriteAverageAndStandardDeviation(results);
        results.Write(m_out_path.c_str());
    }
    
private:
    // C++ formatters
    TimeMeasurement<std::function<void(SBValue)>> m_dump_std_vector_measurement;
    TimeMeasurement<std::function<void(SBValue)>> m_dump_std_list_measurement;
    TimeMeasurement<std::function<void(SBValue)>> m_dump_std_map_measurement;
    TimeMeasurement<std::function<void()>> m_dump_std_string_measurement;

    // Cocoa formatters
    TimeMeasurement<std::function<void(SBValue)>> m_dump_nsstring_measurement;
    TimeMeasurement<std::function<void(SBValue)>> m_dump_nsarray_measurement;
    TimeMeasurement<std::function<void(SBValue)>> m_dump_nsdictionary_measurement;
    TimeMeasurement<std::function<void(SBValue)>> m_dump_nsset_measurement;
    TimeMeasurement<std::function<void(SBValue)>> m_dump_nsbundle_measurement;
    TimeMeasurement<std::function<void(SBValue)>> m_dump_nsdate_measurement;

    // useful files
    std::string m_app_path;
    std::string m_out_path;
};

// argv[1] == path to app
// argv[2] == path to result
int main(int argc, const char * argv[])
{
    FormattersTest frmtest;
    frmtest.SetVerbose(true);
    TestCase::Run(frmtest,argc,argv);
    return 0;
}

