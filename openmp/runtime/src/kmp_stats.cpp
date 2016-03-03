/** @file kmp_stats.cpp
 * Statistics gathering and processing.
 */


//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//

#include "kmp.h"
#include "kmp_str.h"
#include "kmp_lock.h"
#include "kmp_stats.h"

#include <algorithm>
#include <sstream>
#include <iomanip>
#include <stdlib.h>                             // for atexit

#define STRINGIZE2(x) #x
#define STRINGIZE(x) STRINGIZE2(x)

#define expandName(name,flags,ignore)  {STRINGIZE(name),flags},
statInfo timeStat::timerInfo[] = {
    KMP_FOREACH_TIMER(expandName,0)
    {0,0}
};
const statInfo counter::counterInfo[] = {
    KMP_FOREACH_COUNTER(expandName,0)
    {0,0}
};
#undef expandName

#define expandName(ignore1,ignore2,ignore3)  {0.0,0.0,0.0},
kmp_stats_output_module::rgb_color kmp_stats_output_module::timerColorInfo[] = {
    KMP_FOREACH_TIMER(expandName,0)
    {0.0,0.0,0.0}
};
#undef expandName

const kmp_stats_output_module::rgb_color kmp_stats_output_module::globalColorArray[] = {
    {1.0, 0.0, 0.0}, // red
    {1.0, 0.6, 0.0}, // orange
    {1.0, 1.0, 0.0}, // yellow
    {0.0, 1.0, 0.0}, // green 
    {0.0, 0.0, 1.0}, // blue
    {0.6, 0.2, 0.8}, // purple
    {1.0, 0.0, 1.0}, // magenta
    {0.0, 0.4, 0.2}, // dark green
    {1.0, 1.0, 0.6}, // light yellow
    {0.6, 0.4, 0.6}, // dirty purple
    {0.0, 1.0, 1.0}, // cyan
    {1.0, 0.4, 0.8}, // pink
    {0.5, 0.5, 0.5}, // grey
    {0.8, 0.7, 0.5}, // brown
    {0.6, 0.6, 1.0}, // light blue
    {1.0, 0.7, 0.5}, // peach
    {0.8, 0.5, 1.0}, // lavender
    {0.6, 0.0, 0.0}, // dark red
    {0.7, 0.6, 0.0}, // gold
    {0.0, 0.0, 0.0}  // black
};

// Ensure that the atexit handler only runs once.
static uint32_t statsPrinted = 0;

// output interface
static kmp_stats_output_module __kmp_stats_global_output;

/* ****************************************************** */
/* ************* statistic member functions ************* */

void statistic::addSample(double sample)
{
    double delta = sample - meanVal;

    sampleCount = sampleCount + 1;
    meanVal     = meanVal + delta/sampleCount;
    m2          = m2 + delta*(sample - meanVal);

    minVal = std::min(minVal, sample);
    maxVal = std::max(maxVal, sample);
}

statistic & statistic::operator+= (const statistic & other)
{
    if (sampleCount == 0)
    {
        *this = other;
        return *this;
    }

    uint64_t newSampleCount = sampleCount + other.sampleCount;
    double dnsc  = double(newSampleCount);
    double dsc   = double(sampleCount);
    double dscBydnsc = dsc/dnsc;
    double dosc  = double(other.sampleCount);
    double delta = other.meanVal - meanVal;

    // Try to order these calculations to avoid overflows.
    // If this were Fortran, then the compiler would not be able to re-order over brackets.
    // In C++ it may be legal to do that (we certainly hope it doesn't, and CC+ Programming Language 2nd edition
    // suggests it shouldn't, since it says that exploitation of associativity can only be made if the operation
    // really is associative (which floating addition isn't...)).
    meanVal     = meanVal*dscBydnsc + other.meanVal*(1-dscBydnsc);
    m2          = m2 + other.m2 + dscBydnsc*dosc*delta*delta;
    minVal      = std::min (minVal, other.minVal);
    maxVal      = std::max (maxVal, other.maxVal);
    sampleCount = newSampleCount;


    return *this;
}

void statistic::scale(double factor)
{
    minVal = minVal*factor;
    maxVal = maxVal*factor;
    meanVal= meanVal*factor;
    m2     = m2*factor*factor;
    return;
}

std::string statistic::format(char unit, bool total) const
{
    std::string result = formatSI(sampleCount,9,' ');
    
    if (sampleCount == 0)
    {
        result = result + std::string(", ") + formatSI(0.0, 9, unit);
        result = result + std::string(", ") + formatSI(0.0, 9, unit);
        result = result + std::string(", ") + formatSI(0.0, 9, unit);
        if (total)
            result = result + std::string(", ") + formatSI(0.0, 9, unit);
        result = result + std::string(", ") + formatSI(0.0, 9, unit);
    }
    else
    {
        result = result + std::string(", ") + formatSI(minVal,  9, unit);
        result = result + std::string(", ") + formatSI(meanVal, 9, unit);
        result = result + std::string(", ") + formatSI(maxVal,  9, unit);
        if (total)
            result = result + std::string(", ") + formatSI(meanVal*sampleCount, 9, unit);
        result = result + std::string(", ") + formatSI(getSD(), 9, unit);
    }
    return result;
}

/* ********************************************************** */
/* ************* explicitTimer member functions ************* */

void explicitTimer::start(timer_e timerEnumValue) { 
    startTime = tsc_tick_count::now(); 
    if(timeStat::logEvent(timerEnumValue)) {
        __kmp_stats_thread_ptr->incrementNestValue();
    }
    return;
}

void explicitTimer::stop(timer_e timerEnumValue) {
    if (startTime.getValue() == 0)
        return;

    tsc_tick_count finishTime = tsc_tick_count::now();

    //stat->addSample ((tsc_tick_count::now() - startTime).ticks());
    stat->addSample ((finishTime - startTime).ticks());

    if(timeStat::logEvent(timerEnumValue)) {
        __kmp_stats_thread_ptr->push_event(startTime.getValue() - __kmp_stats_start_time.getValue(), finishTime.getValue() - __kmp_stats_start_time.getValue(), __kmp_stats_thread_ptr->getNestValue(), timerEnumValue); 
        __kmp_stats_thread_ptr->decrementNestValue();
    }

    /* We accept the risk that we drop a sample because it really did start at t==0. */
    startTime = 0; 
    return;
}

/* ******************************************************************* */
/* ************* kmp_stats_event_vector member functions ************* */

void kmp_stats_event_vector::deallocate() {
    __kmp_free(events);
    internal_size = 0;
    allocated_size = 0;
    events = NULL;
}

// This function is for qsort() which requires the compare function to return
// either a negative number if event1 < event2, a positive number if event1 > event2
// or zero if event1 == event2.  
// This sorts by start time (lowest to highest).
int compare_two_events(const void* event1, const void* event2) {
    kmp_stats_event* ev1 = (kmp_stats_event*)event1;
    kmp_stats_event* ev2 = (kmp_stats_event*)event2;

    if(ev1->getStart() < ev2->getStart()) return -1;
    else if(ev1->getStart() > ev2->getStart()) return 1;
    else return 0;
}

void kmp_stats_event_vector::sort() {
    qsort(events, internal_size, sizeof(kmp_stats_event), compare_two_events);
}

/* *********************************************************** */
/* ************* kmp_stats_list member functions ************* */

// returns a pointer to newly created stats node
kmp_stats_list* kmp_stats_list::push_back(int gtid) { 
    kmp_stats_list* newnode = (kmp_stats_list*)__kmp_allocate(sizeof(kmp_stats_list));
    // placement new, only requires space and pointer and initializes (so __kmp_allocate instead of C++ new[] is used)
    new (newnode) kmp_stats_list();
    newnode->setGtid(gtid);
    newnode->prev = this->prev;
    newnode->next = this;
    newnode->prev->next = newnode;
    newnode->next->prev = newnode;
    return newnode;
}
void kmp_stats_list::deallocate() {
    kmp_stats_list* ptr = this->next;
    kmp_stats_list* delptr = this->next;
    while(ptr != this) {
        delptr = ptr;
        ptr=ptr->next;
        // placement new means we have to explicitly call destructor.
        delptr->_event_vector.deallocate();
        delptr->~kmp_stats_list();
        __kmp_free(delptr);
    }
}
kmp_stats_list::iterator kmp_stats_list::begin() {
    kmp_stats_list::iterator it;
    it.ptr = this->next;
    return it;
}
kmp_stats_list::iterator kmp_stats_list::end() {
    kmp_stats_list::iterator it;
    it.ptr = this;
    return it;
}
int kmp_stats_list::size() {
    int retval;
    kmp_stats_list::iterator it;
    for(retval=0, it=begin(); it!=end(); it++, retval++) {}
    return retval;
}

/* ********************************************************************* */
/* ************* kmp_stats_list::iterator member functions ************* */

kmp_stats_list::iterator::iterator() : ptr(NULL) {} 
kmp_stats_list::iterator::~iterator() {}
kmp_stats_list::iterator kmp_stats_list::iterator::operator++() {
    this->ptr = this->ptr->next;
    return *this;
}
kmp_stats_list::iterator kmp_stats_list::iterator::operator++(int dummy) {
    this->ptr = this->ptr->next;
    return *this;
}
kmp_stats_list::iterator kmp_stats_list::iterator::operator--() {
    this->ptr = this->ptr->prev;
    return *this;
}
kmp_stats_list::iterator kmp_stats_list::iterator::operator--(int dummy) {
    this->ptr = this->ptr->prev;
    return *this;
}
bool kmp_stats_list::iterator::operator!=(const kmp_stats_list::iterator & rhs) {
   return this->ptr!=rhs.ptr; 
}
bool kmp_stats_list::iterator::operator==(const kmp_stats_list::iterator & rhs) {
   return this->ptr==rhs.ptr; 
}
kmp_stats_list* kmp_stats_list::iterator::operator*() const {
    return this->ptr;
}

/* *************************************************************** */
/* *************  kmp_stats_output_module functions ************** */

const char* kmp_stats_output_module::outputFileName = NULL;
const char* kmp_stats_output_module::eventsFileName = NULL;
const char* kmp_stats_output_module::plotFileName   = NULL;
int kmp_stats_output_module::printPerThreadFlag       = 0;
int kmp_stats_output_module::printPerThreadEventsFlag = 0;

// init() is called very near the beginning of execution time in the constructor of __kmp_stats_global_output
void kmp_stats_output_module::init() 
{
    char * statsFileName  = getenv("KMP_STATS_FILE");
    eventsFileName        = getenv("KMP_STATS_EVENTS_FILE");
    plotFileName          = getenv("KMP_STATS_PLOT_FILE");
    char * threadStats    = getenv("KMP_STATS_THREADS");
    char * threadEvents   = getenv("KMP_STATS_EVENTS");

    // set the stats output filenames based on environment variables and defaults
    outputFileName = statsFileName;
    eventsFileName = eventsFileName ? eventsFileName : "events.dat";
    plotFileName   = plotFileName   ? plotFileName   : "events.plt";

    // set the flags based on environment variables matching: true, on, 1, .true. , .t. , yes
    printPerThreadFlag        = __kmp_str_match_true(threadStats);
    printPerThreadEventsFlag  = __kmp_str_match_true(threadEvents);

    if(printPerThreadEventsFlag) {
        // assigns a color to each timer for printing
        setupEventColors();
    } else {
        // will clear flag so that no event will be logged
        timeStat::clearEventFlags();
    }

    return;
}

void kmp_stats_output_module::setupEventColors() {
    int i;
    int globalColorIndex = 0;
    int numGlobalColors = sizeof(globalColorArray) / sizeof(rgb_color);
    for(i=0;i<TIMER_LAST;i++) {
        if(timeStat::logEvent((timer_e)i)) {
            timerColorInfo[i] = globalColorArray[globalColorIndex];
            globalColorIndex = (globalColorIndex+1)%numGlobalColors;
        }
    }
    return;
}

void kmp_stats_output_module::printStats(FILE *statsOut, statistic const * theStats, bool areTimers)
{
    if (areTimers)
    {
        // Check if we have useful timers, since we don't print zero value timers we need to avoid
        // printing a header and then no data.
        bool haveTimers = false;
        for (int s = 0; s<TIMER_LAST; s++)
        {
            if (theStats[s].getCount() != 0)
            {
                haveTimers = true;
                break;
            }
        }
        if (!haveTimers)
            return;
    }

    // Print
    const char * title = areTimers ? "Timer,                   SampleCount," : "Counter,                 ThreadCount,";
    fprintf (statsOut, "%s    Min,      Mean,       Max,     Total,        SD\n", title);    
    if (areTimers) {
        for (int s = 0; s<TIMER_LAST; s++) {
            statistic const * stat = &theStats[s];
            if (stat->getCount() != 0) {
                char tag = timeStat::noUnits(timer_e(s)) ? ' ' : 'T';
                fprintf (statsOut, "%-25s, %s\n", timeStat::name(timer_e(s)), stat->format(tag, true).c_str());
            }
        }
    } else {   // Counters
        for (int s = 0; s<COUNTER_LAST; s++) {
            statistic const * stat = &theStats[s];
            fprintf (statsOut, "%-25s, %s\n", counter::name(counter_e(s)), stat->format(' ', true).c_str());
        }
    }
} 

void kmp_stats_output_module::printCounters(FILE * statsOut, counter const * theCounters)
{
    // We print all the counters even if they are zero.
    // That makes it easier to slice them into a spreadsheet if you need to.
    fprintf (statsOut, "\nCounter,                    Count\n");
    for (int c = 0; c<COUNTER_LAST; c++) {
        counter const * stat = &theCounters[c];
        fprintf (statsOut, "%-25s, %s\n", counter::name(counter_e(c)), formatSI(stat->getValue(), 9, ' ').c_str());
    }
}

void kmp_stats_output_module::printEvents(FILE* eventsOut, kmp_stats_event_vector* theEvents, int gtid) {
    // sort by start time before printing
    theEvents->sort();
    for (int i = 0; i < theEvents->size(); i++) {
        kmp_stats_event ev = theEvents->at(i);
        rgb_color color = getEventColor(ev.getTimerName());
        fprintf(eventsOut, "%d %lu %lu %1.1f rgb(%1.1f,%1.1f,%1.1f) %s\n", 
                gtid, 
                ev.getStart(), 
                ev.getStop(), 
                1.2 - (ev.getNestLevel() * 0.2),
                color.r, color.g, color.b,
                timeStat::name(ev.getTimerName())
               );
    }
    return;
}

void kmp_stats_output_module::windupExplicitTimers()
{
    // Wind up any explicit timers. We assume that it's fair at this point to just walk all the explcit timers in all threads 
    // and say "it's over".
    // If the timer wasn't running, this won't record anything anyway.
    kmp_stats_list::iterator it;
    for(it = __kmp_stats_list.begin(); it != __kmp_stats_list.end(); it++) {
        for (int timer=0; timer<EXPLICIT_TIMER_LAST; timer++) {
            (*it)->getExplicitTimer(explicit_timer_e(timer))->stop((timer_e)timer);
        }
    }
}

void kmp_stats_output_module::printPloticusFile() {
    int i;
    int size = __kmp_stats_list.size();
    FILE* plotOut = fopen(plotFileName, "w+");

    fprintf(plotOut, "#proc page\n"
                     "   pagesize: 15 10\n"
                     "   scale: 1.0\n\n");

    fprintf(plotOut, "#proc getdata\n"
                     "   file: %s\n\n", 
                     eventsFileName);

    fprintf(plotOut, "#proc areadef\n"
                     "   title: OpenMP Sampling Timeline\n"
                     "   titledetails: align=center size=16\n"
                     "   rectangle: 1 1 13 9\n"
                     "   xautorange: datafield=2,3\n"
                     "   yautorange: -1 %d\n\n", 
                     size);

    fprintf(plotOut, "#proc xaxis\n"
                     "   stubs: inc\n"
                     "   stubdetails: size=12\n"
                     "   label: Time (ticks)\n"
                     "   labeldetails: size=14\n\n");

    fprintf(plotOut, "#proc yaxis\n"
                     "   stubs: inc 1\n"
                     "   stubrange: 0 %d\n"
                     "   stubdetails: size=12\n"
                     "   label: Thread #\n"
                     "   labeldetails: size=14\n\n", 
                     size-1);

    fprintf(plotOut, "#proc bars\n"
                     "   exactcolorfield: 5\n"
                     "   axis: x\n"
                     "   locfield: 1\n"
                     "   segmentfields: 2 3\n"
                     "   barwidthfield: 4\n\n");

    // create legend entries corresponding to the timer color
    for(i=0;i<TIMER_LAST;i++) {
        if(timeStat::logEvent((timer_e)i)) {
            rgb_color c = getEventColor((timer_e)i);
            fprintf(plotOut, "#proc legendentry\n"
                             "   sampletype: color\n"
                             "   label: %s\n"
                             "   details: rgb(%1.1f,%1.1f,%1.1f)\n\n",
                             timeStat::name((timer_e)i),
                             c.r, c.g, c.b);

        }
    }

    fprintf(plotOut, "#proc legend\n"
                     "   format: down\n"
                     "   location: max max\n\n");
    fclose(plotOut);
    return;
}

void kmp_stats_output_module::outputStats(const char* heading) 
{
    statistic allStats[TIMER_LAST];
    statistic allCounters[COUNTER_LAST];

    // stop all the explicit timers for all threads
    windupExplicitTimers();

    FILE * eventsOut;
    FILE * statsOut = outputFileName ? fopen (outputFileName, "a+") : stderr;

    if (eventPrintingEnabled()) {
        eventsOut = fopen(eventsFileName, "w+");
    }

    if (!statsOut)
        statsOut = stderr;

    fprintf(statsOut, "%s\n",heading);
    // Accumulate across threads.
    kmp_stats_list::iterator it;
    for (it = __kmp_stats_list.begin(); it != __kmp_stats_list.end(); it++) {
        int t = (*it)->getGtid();
        // Output per thread stats if requested.
        if (perThreadPrintingEnabled()) {
            fprintf (statsOut, "Thread %d\n", t);
            printStats(statsOut, (*it)->getTimers(), true);
            printCounters(statsOut, (*it)->getCounters());
            fprintf(statsOut,"\n");
        }
        // Output per thread events if requested.
        if (eventPrintingEnabled()) {
            kmp_stats_event_vector events = (*it)->getEventVector();
            printEvents(eventsOut, &events, t);
        }

        for (int s = 0; s<TIMER_LAST; s++) {
            // See if we should ignore this timer when aggregating
            if ((timeStat::masterOnly(timer_e(s)) && (t != 0)) || // Timer is only valid on the master and this thread is a worker
                (timeStat::workerOnly(timer_e(s)) && (t == 0)) || // Timer is only valid on a worker and this thread is the master
                timeStat::synthesized(timer_e(s))                 // It's a synthesized stat, so there's no raw data for it.
               )            
            {
                continue;
            }

            statistic * threadStat = (*it)->getTimer(timer_e(s));
            allStats[s] += *threadStat;
        }

        // Special handling for synthesized statistics.
        // These just have to be coded specially here for now. 
        // At present we only have a few: 
        // The total parallel work done in each thread.
        // The variance here makes it easy to see load imbalance over the whole program (though, of course,
        // it's possible to have a code with awful load balance in every parallel region but perfect load
        // balance oever the whole program.)
        // The time spent in barriers in each thread.
        allStats[TIMER_Total_work].addSample ((*it)->getTimer(TIMER_OMP_work)->getTotal());

        // Time in explicit barriers.
        allStats[TIMER_Total_barrier].addSample ((*it)->getTimer(TIMER_OMP_barrier)->getTotal());

        for (int c = 0; c<COUNTER_LAST; c++) {
            if (counter::masterOnly(counter_e(c)) && t != 0)
                continue;
            allCounters[c].addSample ((*it)->getCounter(counter_e(c))->getValue());
        }
    }

    if (eventPrintingEnabled()) {
        printPloticusFile();
        fclose(eventsOut);
    }

    fprintf (statsOut, "Aggregate for all threads\n");
    printStats (statsOut, &allStats[0], true);
    fprintf (statsOut, "\n");
    printStats (statsOut, &allCounters[0], false);

    if (statsOut != stderr)
        fclose(statsOut);

}

/* ************************************************** */
/* *************  exported C functions ************** */

// no name mangling for these functions, we want the c files to be able to get at these functions
extern "C" {

void __kmp_reset_stats()
{
    kmp_stats_list::iterator it;
    for(it = __kmp_stats_list.begin(); it != __kmp_stats_list.end(); it++) {
        timeStat * timers     = (*it)->getTimers();
        counter * counters    = (*it)->getCounters();
        explicitTimer * eTimers = (*it)->getExplicitTimers();

        for (int t = 0; t<TIMER_LAST; t++)
            timers[t].reset();

        for (int c = 0; c<COUNTER_LAST; c++)
            counters[c].reset();

        for (int t=0; t<EXPLICIT_TIMER_LAST; t++)
            eTimers[t].reset();

        // reset the event vector so all previous events are "erased"
        (*it)->resetEventVector();

        // May need to restart the explicit timers in thread zero?
    }
    KMP_START_EXPLICIT_TIMER(OMP_serial);
    KMP_START_EXPLICIT_TIMER(OMP_start_end);
}

// This function will reset all stats and stop all threads' explicit timers if they haven't been stopped already.
void __kmp_output_stats(const char * heading)
{
    __kmp_stats_global_output.outputStats(heading);
    __kmp_reset_stats();
}

void __kmp_accumulate_stats_at_exit(void)
{
    // Only do this once.
    if (KMP_XCHG_FIXED32(&statsPrinted, 1) != 0)
        return;

    __kmp_output_stats("Statistics on exit");
    return;
}

void __kmp_stats_init(void) 
{
    return;
}

} // extern "C" 

