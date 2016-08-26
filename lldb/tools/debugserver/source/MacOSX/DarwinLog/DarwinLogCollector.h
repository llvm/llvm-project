//===-- DarwinLogCollector.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef DarwinLogCollector_h
#define DarwinLogCollector_h

#include <sys/types.h>

#include <memory>
#include <mutex>
#include <unordered_map>

#include "ActivityStore.h"
#include "ActivityStreamSPI.h"
#include "DarwinLogEvent.h"
#include "DarwinLogInterfaces.h"
#include "DNBDefs.h"
#include "JSON.h"

class DarwinLogCollector;
typedef std::shared_ptr<DarwinLogCollector> DarwinLogCollectorSP;

class DarwinLogCollector:
    public std::enable_shared_from_this<DarwinLogCollector>,
    public ActivityStore
{
public:

    //------------------------------------------------------------------
    /// Return whether the os_log and activity tracing SPI is available.
    ///
    /// @return \b true if the activity stream support is available,
    /// \b false otherwise.
    //------------------------------------------------------------------
    static bool
    IsSupported();

    //------------------------------------------------------------------
    /// Return a log function suitable for DNBLog to use as the internal
    /// logging function.
    ///
    /// @return a DNBLog-style logging function if IsSupported() returns
    ///      true; otherwise, returns nullptr.
    //------------------------------------------------------------------
    static DNBCallbackLog
    GetLogFunction();

    static bool
    StartCollectingForProcess(nub_process_t pid, const JSONObject &config);

    static bool
    CancelStreamForProcess(nub_process_t pid);

    static DarwinLogEventVector
    GetEventsForProcess(nub_process_t pid);

    ~DarwinLogCollector();

    pid_t
    GetProcessID() const
    {
        return m_pid;
    }

    //------------------------------------------------------------------
    // ActivityStore API
    //------------------------------------------------------------------
    const char*
    GetActivityForID(os_activity_id_t activity_id) const override;

    std::string
    GetActivityChainForID(os_activity_id_t activity_id) const override;


private:

    DarwinLogCollector() = delete;
    DarwinLogCollector(const DarwinLogCollector&) = delete;
    DarwinLogCollector &operator=(const DarwinLogCollector&) = delete;

    explicit
    DarwinLogCollector(nub_process_t pid,
                       const LogFilterChainSP &filter_chain_sp);

    void
    SignalDataAvailable();

    void
    SetActivityStream(os_activity_stream_t activity_stream);

    bool
    HandleStreamEntry(os_activity_stream_entry_t entry, int error);

    DarwinLogEventVector
    RemoveEvents();

    void
    CancelActivityStream();

    void
    GetActivityChainForID_internal(os_activity_id_t activity_id,
                                   std::string &result, size_t depth) const;

    struct ActivityInfo
    {
        ActivityInfo(const char *name, os_activity_id_t activity_id,
                     os_activity_id_t parent_activity_id) :
            m_name(name),
            m_id(activity_id),
            m_parent_id(parent_activity_id)
        {
        }

        const std::string       m_name;
        const os_activity_id_t  m_id;
        const os_activity_id_t  m_parent_id;
    };

    using ActivityMap = std::unordered_map<os_activity_id_t, ActivityInfo>;

    const nub_process_t   m_pid;
    os_activity_stream_t  m_activity_stream;
    DarwinLogEventVector  m_events;
    std::mutex            m_events_mutex;
    LogFilterChainSP      m_filter_chain_sp;

    /// Mutex to protect activity info (activity name and parent structures)
    mutable std::mutex    m_activity_info_mutex;
    /// Map of activity id to ActivityInfo
    ActivityMap           m_activity_map;
};

#endif /* LogStreamCollector_h */
