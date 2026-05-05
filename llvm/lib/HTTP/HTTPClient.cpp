//===--- HTTPClient.cpp - HTTP client library -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the implementation of the HTTPClient library for issuing
/// HTTP requests and handling the responses.
///
//===----------------------------------------------------------------------===//

#include "llvm/HTTP/HTTPClient.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#ifdef LLVM_ENABLE_CURL
#include <curl/curl.h>
#endif
#ifdef _WIN32
#include "llvm/Support/ConvertUTF.h"
#endif

using namespace llvm;

HTTPRequest::HTTPRequest(StringRef Url) { this->Url = Url.str(); }

bool operator==(const HTTPRequest &A, const HTTPRequest &B) {
  return A.Url == B.Url && A.Method == B.Method &&
         A.FollowRedirects == B.FollowRedirects;
}

HTTPResponseHandler::~HTTPResponseHandler() = default;

bool HTTPClient::IsInitialized = false;

class HTTPClientCleanup {
public:
  ~HTTPClientCleanup() { HTTPClient::cleanup(); }
};
ManagedStatic<HTTPClientCleanup> Cleanup;

#ifdef LLVM_ENABLE_CURL

bool HTTPClient::isAvailable() { return true; }

void HTTPClient::initialize() {
  if (!IsInitialized) {
    curl_global_init(CURL_GLOBAL_ALL);
    IsInitialized = true;
  }
}

void HTTPClient::cleanup() {
  if (IsInitialized) {
    curl_global_cleanup();
    IsInitialized = false;
  }
}

void HTTPClient::setTimeout(std::chrono::milliseconds Timeout) {
  if (Timeout < std::chrono::milliseconds(0))
    Timeout = std::chrono::milliseconds(0);
  curl_easy_setopt(Handle, CURLOPT_TIMEOUT_MS, Timeout.count());
}

/// CurlHTTPRequest and the curl{Header,Write}Function are implementation
/// details used to work with Curl. Curl makes callbacks with a single
/// customizable pointer parameter.
struct CurlHTTPRequest {
  CurlHTTPRequest(HTTPResponseHandler &Handler) : Handler(Handler) {}
  void storeError(Error Err) {
    ErrorState = joinErrors(std::move(Err), std::move(ErrorState));
  }
  HTTPResponseHandler &Handler;
  llvm::Error ErrorState = Error::success();
};

static size_t curlWriteFunction(char *Contents, size_t Size, size_t NMemb,
                                CurlHTTPRequest *CurlRequest) {
  Size *= NMemb;
  if (Error Err =
          CurlRequest->Handler.handleBodyChunk(StringRef(Contents, Size))) {
    CurlRequest->storeError(std::move(Err));
    return 0;
  }
  return Size;
}

HTTPClient::HTTPClient() {
  assert(IsInitialized &&
         "Must call HTTPClient::initialize() at the beginning of main().");
  if (Handle)
    return;
  Handle = curl_easy_init();
  assert(Handle && "Curl could not be initialized");
  // Set the callback hooks.
  curl_easy_setopt(Handle, CURLOPT_WRITEFUNCTION, curlWriteFunction);
  // Detect supported compressed encodings and accept all.
  curl_easy_setopt(Handle, CURLOPT_ACCEPT_ENCODING, "");
}

HTTPClient::~HTTPClient() { curl_easy_cleanup(Handle); }

Error HTTPClient::perform(const HTTPRequest &Request,
                          HTTPResponseHandler &Handler) {
  if (Request.Method != HTTPMethod::GET)
    return createStringError(errc::invalid_argument,
                             "Unsupported CURL request method.");

  SmallString<128> Url = Request.Url;
  curl_easy_setopt(Handle, CURLOPT_URL, Url.c_str());
  curl_easy_setopt(Handle, CURLOPT_FOLLOWLOCATION, Request.FollowRedirects);

  curl_slist *Headers = nullptr;
  for (const std::string &Header : Request.Headers)
    Headers = curl_slist_append(Headers, Header.c_str());
  curl_easy_setopt(Handle, CURLOPT_HTTPHEADER, Headers);

  CurlHTTPRequest CurlRequest(Handler);
  curl_easy_setopt(Handle, CURLOPT_WRITEDATA, &CurlRequest);
  CURLcode CurlRes = curl_easy_perform(Handle);
  curl_slist_free_all(Headers);
  if (CurlRes != CURLE_OK)
    return joinErrors(std::move(CurlRequest.ErrorState),
                      createStringError(errc::io_error,
                                        "curl_easy_perform() failed: %s\n",
                                        curl_easy_strerror(CurlRes)));
  return std::move(CurlRequest.ErrorState);
}

unsigned HTTPClient::responseCode() {
  long Code = 0;
  curl_easy_getinfo(Handle, CURLINFO_RESPONSE_CODE, &Code);
  return Code;
}

#else

#ifdef _WIN32
#include <windows.h>
#include <winhttp.h>

namespace {

struct WinHTTPSession {
  HINTERNET SessionHandle = nullptr;
  HINTERNET ConnectHandle = nullptr;
  HINTERNET RequestHandle = nullptr;
  DWORD ResponseCode = 0;

  ~WinHTTPSession() {
    if (RequestHandle)
      WinHttpCloseHandle(RequestHandle);
    if (ConnectHandle)
      WinHttpCloseHandle(ConnectHandle);
    if (SessionHandle)
      WinHttpCloseHandle(SessionHandle);
  }
};

bool parseURL(StringRef Url, std::wstring &Host, std::wstring &Path,
              INTERNET_PORT &Port, bool &Secure) {
  // Parse URL: http://host:port/path
  if (Url.starts_with("https://")) {
    Secure = true;
    Url = Url.drop_front(8);
  } else if (Url.starts_with("http://")) {
    Secure = false;
    Url = Url.drop_front(7);
  } else {
    return false;
  }

  size_t SlashPos = Url.find('/');
  StringRef HostPort =
      (SlashPos != StringRef::npos) ? Url.substr(0, SlashPos) : Url;
  StringRef PathPart =
      (SlashPos != StringRef::npos) ? Url.substr(SlashPos) : StringRef("/");

  size_t ColonPos = HostPort.find(':');
  StringRef HostStr =
      (ColonPos != StringRef::npos) ? HostPort.substr(0, ColonPos) : HostPort;

  if (!llvm::ConvertUTF8toWide(HostStr, Host))
    return false;
  if (!llvm::ConvertUTF8toWide(PathPart, Path))
    return false;

  if (ColonPos != StringRef::npos) {
    StringRef PortStr = HostPort.substr(ColonPos + 1);
    Port = static_cast<INTERNET_PORT>(std::stoi(PortStr.str()));
  } else {
    Port = Secure ? INTERNET_DEFAULT_HTTPS_PORT : INTERNET_DEFAULT_HTTP_PORT;
  }

  return true;
}

} // namespace

HTTPClient::HTTPClient() : Handle(new WinHTTPSession()) {}

HTTPClient::~HTTPClient() { delete static_cast<WinHTTPSession *>(Handle); }

bool HTTPClient::isAvailable() { return true; }

void HTTPClient::initialize() {
  if (!IsInitialized) {
    IsInitialized = true;
  }
}

void HTTPClient::cleanup() {
  if (IsInitialized) {
    IsInitialized = false;
  }
}

void HTTPClient::setTimeout(std::chrono::milliseconds Timeout) {
  WinHTTPSession *Session = static_cast<WinHTTPSession *>(Handle);
  if (Session && Session->SessionHandle) {
    DWORD TimeoutMs = static_cast<DWORD>(Timeout.count());
    WinHttpSetOption(Session->SessionHandle, WINHTTP_OPTION_CONNECT_TIMEOUT,
                     &TimeoutMs, sizeof(TimeoutMs));
    WinHttpSetOption(Session->SessionHandle, WINHTTP_OPTION_RECEIVE_TIMEOUT,
                     &TimeoutMs, sizeof(TimeoutMs));
    WinHttpSetOption(Session->SessionHandle, WINHTTP_OPTION_SEND_TIMEOUT,
                     &TimeoutMs, sizeof(TimeoutMs));
  }
}

Error HTTPClient::perform(const HTTPRequest &Request,
                          HTTPResponseHandler &Handler) {
  if (Request.Method != HTTPMethod::GET)
    return createStringError(errc::invalid_argument,
                             "Only GET requests are supported.");
  for (const std::string &Header : Request.Headers)
    if (Header.find("\r") != std::string::npos ||
        Header.find("\n") != std::string::npos) {
      return createStringError(errc::invalid_argument,
                               "Unsafe request can lead to header injection.");
    }

  WinHTTPSession *Session = static_cast<WinHTTPSession *>(Handle);

  // Parse URL
  std::wstring Host, Path;
  INTERNET_PORT Port = 0;
  bool Secure = false;
  if (!parseURL(Request.Url, Host, Path, Port, Secure))
    return createStringError(errc::invalid_argument,
                             "Invalid URL: " + Request.Url);

  // Create session
  Session->SessionHandle =
      WinHttpOpen(L"LLVM-HTTPClient/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                  WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
  if (!Session->SessionHandle)
    return createStringError(errc::io_error, "Failed to open WinHTTP session");

  // Prevent fallback to TLS 1.0/1.1
  DWORD SecureProtocols =
      WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_2 | WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_3;
  if (!WinHttpSetOption(Session->SessionHandle, WINHTTP_OPTION_SECURE_PROTOCOLS,
                        &SecureProtocols, sizeof(SecureProtocols)))
    return createStringError(errc::io_error, "Failed to set secure protocols");

  // Use HTTP/2 if available
  DWORD EnableHttp2 = WINHTTP_PROTOCOL_FLAG_HTTP2;
  WinHttpSetOption(Session->SessionHandle, WINHTTP_OPTION_ENABLE_HTTP_PROTOCOL,
                   &EnableHttp2, sizeof(EnableHttp2));

  // Create connection
  Session->ConnectHandle =
      WinHttpConnect(Session->SessionHandle, Host.c_str(), Port, 0);
  if (!Session->ConnectHandle) {
    return createStringError(errc::io_error,
                             "Failed to connect to host: " + Request.Url);
  }

  // Open request
  DWORD Flags = WINHTTP_FLAG_REFRESH;
  if (Secure)
    Flags |= WINHTTP_FLAG_SECURE;

  Session->RequestHandle = WinHttpOpenRequest(
      Session->ConnectHandle, L"GET", Path.c_str(), nullptr, WINHTTP_NO_REFERER,
      WINHTTP_DEFAULT_ACCEPT_TYPES, Flags);
  if (!Session->RequestHandle)
    return createStringError(errc::io_error, "Failed to open HTTP request");

  if (Secure) {
    // Enforce checks that certificate wasn't revoked.
    DWORD EnableRevocationChecks = WINHTTP_ENABLE_SSL_REVOCATION;
    if (!WinHttpSetOption(Session->RequestHandle, WINHTTP_OPTION_ENABLE_FEATURE,
                          &EnableRevocationChecks,
                          sizeof(EnableRevocationChecks)))
      return createStringError(
          errc::io_error, "Failed to enable certificate revocation checks");

    // Explicitly enforce default validation. This protects against insecure
    // overrides like SECURITY_FLAG_IGNORE_UNKNOWN_CA.
    DWORD SecurityFlags = 0;
    if (!WinHttpSetOption(Session->RequestHandle, WINHTTP_OPTION_SECURITY_FLAGS,
                          &SecurityFlags, sizeof(SecurityFlags)))
      return createStringError(errc::io_error,
                               "Failed to enforce security flags");
  }

  // Add headers
  for (const std::string &Header : Request.Headers) {
    std::wstring WideHeader;
    if (!llvm::ConvertUTF8toWide(Header, WideHeader))
      continue;
    WinHttpAddRequestHeaders(Session->RequestHandle, WideHeader.c_str(),
                             static_cast<DWORD>(WideHeader.length()),
                             WINHTTP_ADDREQ_FLAG_ADD);
  }

  // Send request
  if (!WinHttpSendRequest(Session->RequestHandle, WINHTTP_NO_ADDITIONAL_HEADERS,
                          0, nullptr, 0, 0, 0))
    return createStringError(errc::io_error, "Failed to send HTTP request");

  // Receive response
  if (!WinHttpReceiveResponse(Session->RequestHandle, nullptr))
    return createStringError(errc::io_error, "Failed to receive HTTP response");

  // Get response code
  DWORD CodeSize = sizeof(Session->ResponseCode);
  if (!WinHttpQueryHeaders(Session->RequestHandle,
                           WINHTTP_QUERY_STATUS_CODE |
                               WINHTTP_QUERY_FLAG_NUMBER,
                           WINHTTP_HEADER_NAME_BY_INDEX, &Session->ResponseCode,
                           &CodeSize, nullptr))
    Session->ResponseCode = 0;

  // Read response body
  DWORD BytesAvailable = 0;
  while (WinHttpQueryDataAvailable(Session->RequestHandle, &BytesAvailable)) {
    if (BytesAvailable == 0)
      break;

    std::vector<char> Buffer(BytesAvailable);
    DWORD BytesRead = 0;
    if (!WinHttpReadData(Session->RequestHandle, Buffer.data(), BytesAvailable,
                         &BytesRead))
      return createStringError(errc::io_error, "Failed to read HTTP response");

    if (BytesRead > 0) {
      if (Error Err =
              Handler.handleBodyChunk(StringRef(Buffer.data(), BytesRead)))
        return Err;
    }
  }

  return Error::success();
}

unsigned HTTPClient::responseCode() {
  WinHTTPSession *Session = static_cast<WinHTTPSession *>(Handle);
  return Session ? Session->ResponseCode : 0;
}

#else // _WIN32

// Non-Windows, non-libcurl stub implementations
HTTPClient::HTTPClient() = default;

HTTPClient::~HTTPClient() = default;

bool HTTPClient::isAvailable() { return false; }

void HTTPClient::initialize() {}

void HTTPClient::cleanup() {}

void HTTPClient::setTimeout(std::chrono::milliseconds Timeout) {}

Error HTTPClient::perform(const HTTPRequest &Request,
                          HTTPResponseHandler &Handler) {
  llvm_unreachable("No HTTP Client implementation available.");
}

unsigned HTTPClient::responseCode() {
  llvm_unreachable("No HTTP Client implementation available.");
}

#endif // _WIN32

#endif
