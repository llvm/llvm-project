// The number of seconds to wait at the end of the test inferior before
// exiting.  This delay is needed to ensure the logging infrastructure
// has flushed out the message.  If we finished before all messages were
// flushed, then the test will never see the unflushed messages, causing
// some test logic to fail.
#define FINAL_WAIT_SECONDS 5
