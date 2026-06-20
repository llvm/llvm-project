/* ============================================================
   LLVM Advisor — Application Core (Icons, API, State, Router,
   Theme, Syntax Highlight, Fuzzy Search, Retry)
   ============================================================ */

// --- Icon Set (inline SVG, 20x20) ---
const Icons = {
  overview: `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="6" height="6" rx="1"/><rect x="11" y="3" width="6" height="6" rx="1"/><rect x="3" y="11" width="6" height="6" rx="1"/><rect x="11" y="11" width="6" height="6" rx="1"/></svg>`,
  units: `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5"><line x1="3" y1="5" x2="17" y2="5"/><line x1="3" y1="10" x2="17" y2="10"/><line x1="3" y1="15" x2="17" y2="15"/></svg>`,
  compare: `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="3,14 7,6 11,12 15,4 17,8"/></svg>`,
  timeline: `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="10" cy="10" r="7"/><polyline points="10,6 10,10 13,12"/></svg>`,
  insights: `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="10,2 12,8 18,8 13,12 15,18 10,14 5,18 7,12 2,8 8,8"/></svg>`,
  settings: `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="10" cy="10" r="3"/><path d="M10,2v3M10,15v3M2,10h3M15,10h3M4.2,4.2l2.1,2.1M13.7,13.7l2.1,2.1M4.2,15.8l2.1-2.1M13.7,6.3l2.1-2.1"/></svg>`,
  search: `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="8.5" cy="8.5" r="5"/><line x1="12.5" y1="12.5" x2="17" y2="17"/></svg>`,
  chevronDown: `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2"><polyline points="5,8 10,13 15,8"/></svg>`,
  chevronRight: `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2"><polyline points="8,5 13,10 8,15"/></svg>`,
  pin: `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5"><line x1="10" y1="2" x2="10" y2="14"/><circle cx="10" cy="16" r="2"/></svg>`,
  close: `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2"><line x1="5" y1="5" x2="15" y2="15"/><line x1="15" y1="5" x2="5" y2="15"/></svg>`,
  warning: `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M10,3 L18,17 H2 Z"/><line x1="10" y1="8" x2="10" y2="12"/><circle cx="10" cy="14.5" r=".5" fill="currentColor"/></svg>`,
  error: `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="10" cy="10" r="7"/><line x1="7" y1="7" x2="13" y2="13"/><line x1="13" y1="7" x2="7" y2="13"/></svg>`,
  folder: `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M2,5 V16 H18 V7 H10 L8,5 Z"/></svg>`,
  file: `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4,2 H12 L16,6 V18 H4 Z"/><polyline points="12,2 12,6 16,6"/></svg>`,
  arrowUp: `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2"><line x1="10" y1="16" x2="10" y2="4"/><polyline points="5,9 10,4 15,9"/></svg>`,
  arrowDown: `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2"><line x1="10" y1="4" x2="10" y2="16"/><polyline points="5,11 10,16 15,11"/></svg>`,
  sun: `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="10" cy="10" r="4"/><line x1="10" y1="1" x2="10" y2="3"/><line x1="10" y1="17" x2="10" y2="19"/><line x1="1" y1="10" x2="3" y2="10"/><line x1="17" y1="10" x2="19" y2="10"/><line x1="3.6" y1="3.6" x2="5" y2="5"/><line x1="15" y1="15" x2="16.4" y2="16.4"/><line x1="3.6" y1="16.4" x2="5" y2="15"/><line x1="15" y1="5" x2="16.4" y2="3.6"/></svg>`,
  moon: `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M17 10.5C17 14.0899 14.0899 17 10.5 17C6.91015 17 4 14.0899 4 10.5C4 6.91015 6.91015 4 10.5 4C10.5 4 8 6 8 8.5C8 11 10 13 12.5 13C15 13 17 10.5 17 10.5Z"/></svg>`,
};

// --- Security Utilities ---
const escapeHtml = (str) => {
  if (str == null) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
};

// Render a trusted inline SVG icon string into a span element.
const renderIcon = (svgString) => {
  const span = document.createElement('span');
  span.className = 'icon';
  span.innerHTML = svgString;
  return span;
};

// --- Theme ---
const Theme = {
  init() {
    const saved = localStorage.getItem('advisor-theme');
    this.apply(saved || 'auto');
  },
  toggle() {
    const next = this.isDark() ? 'light' : 'dark';
    this.apply(next);
    localStorage.setItem('advisor-theme', next);
  },
  apply(mode) {
    document.documentElement.classList.toggle('dark', mode === 'dark');
    document.documentElement.classList.toggle('light', mode === 'light');
  },
  isDark() {
    if (document.documentElement.classList.contains('dark')) return true;
    if (document.documentElement.classList.contains('light')) return false;
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches || false;
  },
  icon() { return this.isDark() ? 'sun' : 'moon'; },
};

// --- API Client with Retry ---
const API = {
  base: '/api/v1',
  async _fetchWithRetry(url, options = {}, retries = 2) {
    let lastErr;
    for (let i = 0; i <= retries; i++) {
      try {
        const r = await fetch(url, options);
        if (!r.ok) {
          const text = await r.text().catch(() => '');
          lastErr = `HTTP ${r.status}: ${text.slice(0, 120)}`;
          if (r.status >= 500 && i < retries) { await new Promise(r => setTimeout(r, 300 * (i + 1))); continue; }
          return { ok: false, error: lastErr, data: null };
        }
        const j = await r.json();
        if (j.status === 'error') return { ok: false, error: j.error?.message || 'Unknown error', data: null };
        return { ok: true, data: j.data ?? j, error: null };
      } catch (e) {
        lastErr = e.message;
        if (i < retries) await new Promise(r => setTimeout(r, 300 * (i + 1)));
      }
    }
    return { ok: false, error: lastErr, data: null };
  },
  async get(path) {
    return this._fetchWithRetry(this.base + path);
  },
  async post(path, body) {
    return this._fetchWithRetry(this.base + path, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
    });
  },
  health: () => API.get('/health'),
  snapshots: () => API.get('/snapshots'),
  snapshot: (id) => API.get(`/snapshots/${id}`),
  snapshotSummary: (id) => API.get(`/snapshots/${id}/summary`),
  units: (snapId) => API.get(`/snapshots/${snapId}/units`),
  unit: (snapId, unitId) => API.get(`/snapshots/${snapId}/units/${unitId}`),
  capabilities: () => API.get('/capabilities'),
  queryUnit: (unitId, caps) => API.get(`/query/unit/${encodeURIComponent(unitId)}/${(caps || []).join(',')}`),
  querySnapshot: (snapshotId, caps) => API.get(`/query/snapshot/${encodeURIComponent(snapshotId)}/${(caps || []).join(',')}`),
  insights: (snapId) => API.get(`/snapshots/${snapId}/insights`),
  insight: (snapId, name, baseline) => {
    let url = `/snapshots/${snapId}/insights/${name}`;
    if (baseline) url += `?baseline=${encodeURIComponent(baseline)}`;
    return API.get(url);
  },
  compare: (before, after) => API.get(`/compare/${encodeURIComponent(before)}/${encodeURIComponent(after)}`),
  inspect: (mode, body) => API.post(`/inspect/${encodeURIComponent(mode)}`, body),
  jobs: () => API.get('/jobs'),
};

// --- State ---
const State = {
  _listeners: new Map(),
  _data: {
    route: '/',
    routeParams: {},
    snapshots: [],
    currentSnapshot: null,
    currentProject: null,
    units: [],
    health: null,
    sidebarPinned: false,
    detailOpen: false,
    detailContent: null,
    commandPaletteOpen: false,
  },
  get(key) { return this._data[key]; },
  set(key, value) {
    this._data[key] = value;
    (this._listeners.get(key) || []).forEach(fn => fn(value));
    (this._listeners.get('*') || []).forEach(fn => fn(key, value));
  },
  on(key, fn) {
    if (!this._listeners.has(key)) this._listeners.set(key, []);
    this._listeners.get(key).push(fn);
  },
};

// --- Router ---
const Router = {
  routes: {},
  register(path, handler) { this.routes[path] = handler; },
  init() {
    window.addEventListener('hashchange', () => this.resolve());
    this.resolve();
  },
  navigate(path) {
    window.location.hash = '#' + path;
  },
  resolve() {
    const hash = window.location.hash.slice(1) || '/';
    const [path, query] = hash.split('?');
    const params = Object.fromEntries(new URLSearchParams(query || ''));

    for (const [pattern, handler] of Object.entries(this.routes)) {
      const match = this._match(pattern, path);
      if (match !== null) {
        State.set('routeParams', { ...match, ...params });
        State.set('route', pattern);
        handler({ ...match, ...params });
        return;
      }
    }
    if (this.routes['/']) {
      State.set('route', '/');
      State.set('routeParams', params);
      this.routes['/'](params);
    }
  },
  _match(pattern, path) {
    const patternParts = pattern.split('/').filter(Boolean);
    const pathParts = path.split('/').filter(Boolean);
    if (patternParts.length !== pathParts.length) return null;
    const params = {};
    for (let i = 0; i < patternParts.length; i++) {
      if (patternParts[i].startsWith(':')) {
        params[patternParts[i].slice(1)] = pathParts[i];
      } else if (patternParts[i] !== pathParts[i]) {
        return null;
      }
    }
    return params;
  },
};

// --- Keyboard Manager ---
const Keys = {
  _pending: null,
  _timeout: null,
  init() {
    document.addEventListener('keydown', e => this._handle(e));
  },
  _handle(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
      if (e.key === 'Escape') e.target.blur();
      return;
    }
    if (e.key === '?' || (e.metaKey && e.key === 'k') || (e.ctrlKey && e.key === 'k')) {
      e.preventDefault();
      State.set('commandPaletteOpen', !State.get('commandPaletteOpen'));
      return;
    }
    if (e.key === 'Escape') {
      if (State.get('commandPaletteOpen')) { State.set('commandPaletteOpen', false); return; }
      if (State.get('detailOpen')) { State.set('detailOpen', false); return; }
      return;
    }
    if (this._pending === 'g') {
      clearTimeout(this._timeout);
      this._pending = null;
      const navMap = { o: '/', u: '/units', c: '/compare', t: '/timeline', i: '/insights', s: '/settings' };
      if (navMap[e.key]) { e.preventDefault(); Router.navigate(navMap[e.key]); }
      return;
    }
    if (e.key === 'g') {
      this._pending = 'g';
      this._timeout = setTimeout(() => { this._pending = null; }, 500);
      return;
    }
  },
};

// --- DOM Helpers ---
const h = (tag, attrs = {}, ...children) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k.startsWith('on')) el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (v === true) el.setAttribute(k, '');
    else el.setAttribute(k, v);
  }
  for (const child of children.flat()) {
    if (child == null) continue;
    if (Array.isArray(child)) child.forEach(c => c != null && el.appendChild(typeof c === 'string' || typeof c === 'number' ? document.createTextNode(String(c)) : c));
    else el.appendChild(typeof child === 'string' || typeof child === 'number' ? document.createTextNode(String(child)) : child);
  }
  return el;
};

const appendValue = (parent, value) => {
  if (value == null || value === false) return;
  if (Array.isArray(value)) { value.forEach(v => appendValue(parent, v)); return; }
  if (value instanceof Node) { parent.appendChild(value); return; }
  parent.appendChild(document.createTextNode(String(value)));
};

const html = (strings, ...values) => {
  const marker = i => `__LLVM_ADVISOR_SLOT_${i}__`;
  const source = strings.reduce((out, s, i) => out + s + (i < values.length ? marker(i) : ''), '');
  const tpl = document.createElement('template');
  tpl.innerHTML = source.trim();

  tpl.content.querySelectorAll('*').forEach(el => {
    for (const attr of Array.from(el.attributes)) {
      const idx = values.findIndex((_, i) => attr.value === marker(i));
      if (idx < 0) continue;
      const value = values[idx];
      if (attr.name.startsWith('@')) {
        if (typeof value === 'function') el.addEventListener(attr.name.slice(1), value);
        el.removeAttribute(attr.name);
      } else {
        el.removeAttribute(attr.name);
        if (value != null && value !== false) el.setAttribute(attr.name, value === true ? '' : String(value));
      }
    }
  });

  const walker = document.createTreeWalker(tpl.content, NodeFilter.SHOW_TEXT);
  const textNodes = [];
  while (walker.nextNode()) textNodes.push(walker.currentNode);
  textNodes.forEach(node => {
    const text = node.nodeValue;
    let cursor = 0;
    const frag = document.createDocumentFragment();
    const re = /__LLVM_ADVISOR_SLOT_(\d+)__/g;
    let match;
    while ((match = re.exec(text))) {
      if (match.index > cursor) frag.appendChild(document.createTextNode(text.slice(cursor, match.index)));
      appendValue(frag, values[Number(match[1])]);
      cursor = match.index + match[0].length;
    }
    if (cursor === 0) return;
    if (cursor < text.length) frag.appendChild(document.createTextNode(text.slice(cursor)));
    node.replaceWith(frag);
  });

  return tpl.content.childNodes.length === 1 ? tpl.content.firstChild : tpl.content;
};

const clearEl = el => { while (el.firstChild) el.removeChild(el.firstChild); };

const formatNumber = n => {
  if (n == null) return '–';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
};

const formatBytes = b => {
  if (b == null) return '–';
  if (b >= 1e9) return (b / 1e9).toFixed(1) + 'GB';
  if (b >= 1e6) return (b / 1e6).toFixed(1) + 'MB';
  if (b >= 1e3) return (b / 1e3).toFixed(1) + 'KB';
  return b + 'B';
};

const timeAgo = ts => {
  if (!ts) return '';
  const now = Date.now() / 1000;
  const diff = now - ts;
  if (diff < 60) return 'just now';
  if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
  if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
  if (diff < 604800) return Math.floor(diff / 86400) + 'd ago';
  return new Date(ts * 1000).toLocaleDateString();
};

const titleCase = s => String(s || '')
  .replace(/[_\-.]+/g, ' ')
  .replace(/\b\w/g, c => c.toUpperCase());

const isCorruptedString = (s) => {
  if (typeof s !== 'string' || !s) return true;
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    if (c === 0xFFFD) return true;
    if (c < 0x20 && c !== 0x09 && c !== 0x0A && c !== 0x0D) return true;
  }
  return false;
};

const capabilityFriendlyNames = {
  'llvm.ir.summary': 'IR Summary',
  'llvm.ir.function_stats': 'IR Function Stats',
  'llvm.ir.view': 'IR View',
  'llvm.ir.diff': 'IR Diff',
  'llvm.ir.passes.list': 'Pass Pipeline',
  'clang.diag.summary': 'Diagnostics',
  'clang.template_stats': 'Template Stats',
  'clang.static_analysis': 'Static Analysis',
  'llvm.remarks.summary': 'Optimization Remarks',
  'llvm.remarks.detail': 'Remark Details',
  'llvm.remarks.size_diff': 'Remark Size Diff',
  'llvm.obj.summary': 'Binary Summary',
  'llvm.obj.sections': 'Binary Sections',
  'llvm.obj.symbols': 'Binary Symbols',
  'llvm.debug.detail': 'Debug Info',
  'llvm.debug.summary': 'Debug Summary',
  'llvm.cfg': 'Control Flow Graph',
  'llvm.dom_tree': 'Dominator Tree',
  'llvm.call_graph': 'Call Graph',
  'llvm.loop_info': 'Loop Info',
  'llvm.selection_dag': 'Selection DAG',
  'llvm.machine_ir': 'Machine IR',
  'llvm.asm.view': 'Assembly',
  'llvm.mca.report': 'Machine Code Analyzer',
  'llvm.exegesis': 'Instruction Benchmarks',
  'llvm.lto.summary': 'LTO Summary',
  'llvm.lto.function_stats': 'LTO Function Stats',
  'llvm.cgdata': 'CG Data',
  'lld.mapfile': 'Linker Map',
  'lld.mapfile.diff': 'Linker Map Diff',
  'offload.binary.inspect': 'Offload Binary',
  'runtime.correlate': 'Runtime Correlation',
  'runtime.summary': 'Runtime Summary',
  'build.compile_commands': 'Compile Commands',
};

const friendlyCapabilityName = (id, fallback) => {
  if (capabilityFriendlyNames[id]) return capabilityFriendlyNames[id];
  if (fallback) return fallback;
  return titleCase(id);
};

// --- Fuzzy Search ---
const fuzzyMatch = (query, text) => {
  if (!query) return { score: 1, matches: [] };
  const q = query.toLowerCase();
  const t = text.toLowerCase();
  let qi = 0, ti = 0, score = 0, matches = [];
  while (qi < q.length && ti < t.length) {
    if (q[qi] === t[ti]) {
      if (ti === 0 || t[ti - 1] === '/' || t[ti - 1] === '_' || t[ti - 1] === '-') score += 3;
      else score += 1;
      matches.push(ti);
      qi++;
    }
    ti++;
  }
  if (qi < q.length) return null;
  // Penalize length difference
  score -= (t.length - q.length) * 0.1;
  return { score, matches };
};

// --- Syntax Highlighting (lightweight inline) ---
const Syntax = {
  highlightLLVMIR(text) {
    return this._tokenize(text, [
      { re: /;.*$/, cls: 'sh-comment' },
      { re: /\b(define|declare|attributes|target|module|source_filename)\b/, cls: 'sh-keyword' },
      { re: /\b(i\d+|half|bfloat|float|double|fp128|x86_fp80|ppc_fp128|x86_mmx|void|label|metadata|token|ptr)\b/, cls: 'sh-type' },
      { re: /\b(alloca|load|store|getelementptr|insertvalue|extractvalue|icmp|fcmp|phi|select|call|invoke|ret|br|switch|indirectbr|resume|unreachable|landingpad|catchpad|cleanuppad|add|fadd|sub|fsub|mul|fmul|udiv|sdiv|fdiv|urem|srem|frem|shl|lshr|ashr|and|or|xor)\b/, cls: 'sh-function' },
      { re: /@[\w.$-]+/, cls: 'sh-label' },
      { re: /%[\w.$-]+/, cls: 'sh-operator' },
      { re: /-?\d+\.?\d*/, cls: 'sh-number' },
      { re: /"(?:\\.|[^"])*"/, cls: 'sh-string' },
    ]);
  },
  highlightAsm(text) {
    return this._tokenize(text, [
      { re: /[#;].*$/, cls: 'sh-comment' },
      { re: /^\s*\.[a-zA-Z]+/, cls: 'sh-directive' },
      { re: /^[a-zA-Z_][\w.]*:/, cls: 'sh-label' },
      { re: /\b(mov|push|pop|lea|call|ret|jmp|je|jne|jg|jl|jge|jle|ja|jb|add|sub|mul|div|and|or|xor|not|neg|inc|dec|shl|shr|sar|cmp|test|nop|int|syscall|leave|enter|ld|st)\b/, cls: 'sh-keyword' },
      { re: /\b(rax|rbx|rcx|rdx|rsi|rdi|rbp|rsp|r8|r9|r10|r11|r12|r13|r14|r15|eax|ebx|ecx|edx|esi|edi|ebp|esp|ax|bx|cx|dx|al|bl|cl|dl|ah|bh|ch|dh|xmm\d+|ymm\d+)\b/, cls: 'sh-type' },
      { re: /-?\d+\.?\d*/, cls: 'sh-number' },
      { re: /"(?:\\.|[^"])*"/, cls: 'sh-string' },
    ]);
  },
  highlightC(text) {
    return this._tokenize(text, [
      { re: /\/\/.*$|\/\*[\s\S]*?\*\//, cls: 'sh-comment' },
      { re: /\b(int|char|float|double|void|long|short|signed|unsigned|struct|union|enum|typedef|static|extern|inline|const|volatile|restrict|auto|register|sizeof|typeof|bool|true|false|nullptr|NULL)\b/, cls: 'sh-type' },
      { re: /\b(if|else|for|while|do|switch|case|default|break|continue|return|goto|try|catch|throw|new|delete|class|public|private|protected|virtual|override|template|namespace|using|friend|operator|explicit|implicit|constexpr|consteval|constinit|noexcept|decltype|typename|this|super)\b/, cls: 'sh-keyword' },
      { re: /\b[a-zA-Z_][\w]*\s*(?=\()/, cls: 'sh-function' },
      { re: /-?\d+\.?\d*[ulULfF]*\b/, cls: 'sh-number' },
      { re: /"(?:\\.|[^"])*"|'(?:\\.|[^'])*'/, cls: 'sh-string' },
      { re: /\#[ \t]*[a-zA-Z]+/, cls: 'sh-directive' },
    ]);
  },
  _tokenize(text, rules) {
    const lines = text.split('\n');
    return lines.map(line => {
      const parts = [];
      let pos = 0;
      while (pos < line.length) {
        let matched = false;
        for (const rule of rules) {
          rule.re.lastIndex = 0;
          const m = line.slice(pos).match(rule.re);
          if (m && m.index === 0) {
            parts.push(h('span', { class: rule.cls }, m[0]));
            pos += m[0].length;
            matched = true;
            break;
          }
        }
        if (!matched) {
          parts.push(document.createTextNode(line[pos]));
          pos++;
        }
      }
      const pre = h('pre', { class: 'code-line', style: { margin: 0, padding: '2px 12px' } });
      parts.forEach(p => pre.appendChild(typeof p === 'string' ? document.createTextNode(p) : p));
      return pre;
    });
  },
  codeBlock(text, lang) {
    const wrap = h('div', { class: 'code-content', style: { padding: '8px 0' } });
    if (lang === 'ir' || lang === 'llvm') {
      this.highlightLLVMIR(text).forEach(el => wrap.appendChild(el));
    } else if (lang === 'asm' || lang === 's') {
      this.highlightAsm(text).forEach(el => wrap.appendChild(el));
    } else if (lang === 'c' || lang === 'cpp' || lang === 'cc') {
      this.highlightC(text).forEach(el => wrap.appendChild(el));
    } else {
      wrap.appendChild(h('pre', { class: 'raw-json' }, text));
    }
    return wrap;
  },
};

// --- Capability Data Processing ---
const scalarMetricKeys = new Set([
  'errors', 'warnings', 'notes', 'count', 'remark_count', 'function_count', 'functions',
  'instructions', 'instruction_count', 'globals', 'global_count', 'sections', 'symbols', 'symbol_count',
  'compile_units', 'max_depth', 'total_headers', 'total_functions',
  'trace_event_count', 'duration_us', 'wall_time_ns', 'basic_blocks',
  'call_count', 'intrinsic_call_count', 'phi_count', 'memory_op_count',
  'kernel_count', 'transfer_count', 'sync_count', 'bytes', 'size'
]);

const preferredMetricOrder = [
  'instructions', 'instruction_count', 'functions', 'function_count',
  'warnings', 'errors', 'remarks', 'remark_count', 'symbols', 'symbol_count',
  'sections', 'compile_units', 'total_headers', 'trace_event_count',
  'duration_us', 'wall_time_ns', 'memory_op_count', 'call_count',
  'basic_blocks', 'global_count'
];

const ignoredMetricKeys = new Set([
  'available', 'capability', 'unit_id', 'snapshot_id', 'source_path',
  'directory', 'reason', 'summary', 'module', 'remarks_path', 'object_path',
  'ir_path', 'kind', 'format', 'arch', 'tool', 'input', 'stdout', 'stderr',
  'note', 'version'
]);

const CapabilityData = {
  category(capability) {
    const id = String(capability || '');
    if (id.startsWith('build.')) return 'Build';
    if (id.startsWith('clang.')) return 'Clang';
    if (id.startsWith('llvm.ir.') || id === 'llvm.inlining.tree' || id.startsWith('llvm.remarks.') || id.startsWith('llvm.pass.')) return 'IR';
    if (id.startsWith('llvm.obj.') || id.startsWith('llvm.debug.') || id === 'llvm.cgdata' || id.startsWith('lld.mapfile')) return 'Binary';
    if (id.startsWith('llvm.cfg') || id.startsWith('llvm.dom_tree') || id.startsWith('llvm.call_graph') || id.startsWith('llvm.loop_info') || id.startsWith('llvm.selection_dag') || id.startsWith('llvm.machine_ir') || id.startsWith('llvm.asm.') || id.startsWith('llvm.mca.') || id === 'llvm.exegesis') return 'Inspection';
    if (id.startsWith('llvm.lto.')) return 'LTO';
    if (id.startsWith('offload.')) return 'Offload';
    if (id.startsWith('runtime.')) return 'Runtime';
    return 'Other';
  },

  shouldQueryCapability(spec, scope = 'unit') {
    const id = spec?.id || spec?.capability_id || '';
    if (!id) return false;
    if (id === 'llvm.exegesis') return false;
    if (id === 'clang.template_stats' || id === 'clang.static_analysis') return false;
    if (id.startsWith('runtime.')) return false;
    if (id === 'llvm.ir.diff' || id === 'llvm.remarks.size_diff' || id === 'lld.mapfile.diff')
      return scope === 'compare';
    return true;
  },

  isAvailable(value) {
    if (!value || typeof value !== 'object') return false;
    return value.available !== false;
  },

  normalizeResults(results) {
    return (Array.isArray(results) ? results : []).map(result => {
      const value = result?.value && typeof result.value === 'object' ? result.value : {};
      return {
        capability: result?.capability || value.capability || 'unknown',
        cache: result?.cache || '',
        resultId: result?.result_id || '',
        value,
        available: this.isAvailable(value),
        reason: value.reason || result?.error || '',
        metrics: this.metrics(value),
        tables: this.tables(value),
        findings: this.findings(value),
        artifacts: this.artifacts(value),
      };
    });
  },

  metrics(value) {
    const metrics = {};
    if (!value || typeof value !== 'object') return metrics;
    for (const [key, raw] of Object.entries(value)) {
      if (ignoredMetricKeys.has(key)) continue;
      if (typeof raw === 'number' && (scalarMetricKeys.has(key) || !Array.isArray(raw)))
        metrics[key] = raw;
      else if (typeof raw === 'boolean')
        metrics[key] = raw ? 'yes' : 'no';
    }
    if (value.by_type && typeof value.by_type === 'object') {
      for (const [key, raw] of Object.entries(value.by_type))
        if (typeof raw === 'number') metrics[`type_${key}`] = raw;
    }
    return metrics;
  },

  tables(value) {
    const tables = [];
    if (!value || typeof value !== 'object') return tables;
    for (const [key, raw] of Object.entries(value)) {
      if (Array.isArray(raw) && raw.length && raw.every(item => item && typeof item === 'object'))
        tables.push({ name: key, rows: raw });
    }
    return tables;
  },

  findings(value) {
    if (!value || typeof value !== 'object') return [];
    if (Array.isArray(value.findings)) return value.findings;
    if (Array.isArray(value.diagnostics)) {
      return value.diagnostics.map(d => ({
        severity: d.level || d.severity || 'info',
        message: d.message || '',
        file: d.file || '',
        line: d.line,
        column: d.column,
        kind: 'diagnostic',
      }));
    }
    if (Array.isArray(value.remarks)) {
      return value.remarks.map(r => ({
        severity: r.type || 'remark',
        message: r.message || r.name || '',
        pass: r.pass,
        function: r.function,
        file: r.location?.file,
        line: r.location?.line,
        column: r.location?.column,
        hotness: r.hotness,
        kind: 'remark',
      }));
    }
    return [];
  },

  artifacts(value) {
    const artifacts = [];
    if (!value || typeof value !== 'object') return artifacts;
    for (const key of ['stdout', 'stderr', 'ir', 'assembly', 'preprocessed_source', 'text'])
      if (typeof value[key] === 'string' && value[key]) artifacts.push({ name: key, text: value[key] });
    if (value.traceEvents) artifacts.push({ name: 'traceEvents', data: value.traceEvents });
    return artifacts;
  },

  aggregate(unitResults) {
    const agg = {
      units: 0,
      instructions: 0,
      functions: 0,
      warnings: 0,
      errors: 0,
      remarks: 0,
      sections: 0,
      symbols: 0,
      unavailable: 0,
      metrics: {},
      capabilityCoverage: new Map(),
      familyCoverage: new Map(),
    };
    const rows = [];
    (Array.isArray(unitResults) ? unitResults : []).forEach(unit => {
      agg.units++;
      const row = { unit_id: unit.unit_id, source_path: unit.source_path, metrics: {}, results: unit.results || [] };
      this.normalizeResults(unit.results || []).forEach(result => {
        const family = this.category(result.capability);
        const capEntry = agg.capabilityCoverage.get(result.capability) || {
          capability: result.capability,
          family,
          queried: 0,
          available: 0,
          missing: 0,
          reason: '',
          metrics: {},
        };
        capEntry.queried++;
        if (result.available) capEntry.available++;
        else {
          agg.unavailable++;
          capEntry.missing++;
          if (!capEntry.reason) capEntry.reason = result.reason || '';
        }
        agg.capabilityCoverage.set(result.capability, capEntry);

        const familyEntry = agg.familyCoverage.get(family) || {
          family,
          queried: 0,
          available: 0,
          missing: 0,
          metrics: {},
        };
        familyEntry.queried++;
        if (result.available) familyEntry.available++;
        else familyEntry.missing++;
        agg.familyCoverage.set(family, familyEntry);

        const v = result.value;
        const trackExplicitly = new Set([
          'instructions', 'instruction_count', 'functions', 'function_count',
          'warnings', 'errors', 'remarks', 'remark_count',
        ]);
        if (result.available) {
          Object.entries(result.metrics || {}).forEach(([key, raw]) => {
            if (typeof raw !== 'number' || !Number.isFinite(raw)) return;
            agg.metrics[key] = (agg.metrics[key] || 0) + raw;
            capEntry.metrics[key] = (capEntry.metrics[key] || 0) + raw;
            familyEntry.metrics[key] = (familyEntry.metrics[key] || 0) + raw;
            if (!trackExplicitly.has(key)) {
              row.metrics[key] = (row.metrics[key] || 0) + raw;
              row[key] = row.metrics[key];
            }
          });
        }
        if (result.capability === 'llvm.ir.summary') {
          const inst = Number(v.instructions || v.instruction_count || 0);
          const fns = Number(v.functions || v.function_count || 0);
          agg.instructions += inst;
          agg.functions += fns;
          row.instructions = inst || row.instructions || 0;
          row.functions = fns || row.functions || 0;
        }
        if (result.capability === 'llvm.ir.function_stats') {
          const total = Array.isArray(v.functions) ? v.functions.reduce((s, f) => s + Number(f.instructions || f.instruction_count || 0), 0) : 0;
          row.instructions = row.instructions || total;
          agg.instructions += row.instructions && !(v.instructions || v.instruction_count) ? 0 : Number(v.instructions || v.instruction_count || 0);
        }
        if (result.capability === 'clang.diag.summary') {
          agg.warnings += Number(v.warnings || 0);
          agg.errors += Number(v.errors || 0);
          row.warnings = Number(v.warnings || 0);
          row.errors = Number(v.errors || 0);
        }
        if (result.capability === 'llvm.remarks.summary') {
          const cnt = Number(v.count || v.remark_count || 0);
          agg.remarks += cnt;
          row.remarks = cnt;
        }
        if (result.capability === 'llvm.obj.summary') {
          agg.sections += Number(v.sections || 0);
          agg.symbols += Number(v.symbols || v.symbol_count || 0);
          row.sections = Number(v.sections || 0);
          row.symbols = Number(v.symbols || v.symbol_count || 0);
        }
      });
      rows.push(row);
    });
    agg.rows = rows;
    // Override metrics with correctly-tracked per-capability values to avoid
    // double-counting (e.g. 'functions' from IR vs debug/AST capabilities).
    if (agg.instructions) agg.metrics.instructions = agg.instructions;
    if (agg.functions) agg.metrics.functions = agg.functions;
    if (agg.remarks) agg.metrics.remarks = agg.remarks;
    if (agg.warnings) agg.metrics.warnings = agg.warnings;
    if (agg.errors) agg.metrics.errors = agg.errors;
    delete agg.metrics.instruction_count;
    delete agg.metrics.function_count;
    delete agg.metrics.remark_count;
    agg.capabilities = Array.from(agg.capabilityCoverage.values()).sort((a, b) =>
      a.family === b.family ? a.capability.localeCompare(b.capability) : a.family.localeCompare(b.family)
    );
    agg.families = Array.from(agg.familyCoverage.values()).sort((a, b) => a.family.localeCompare(b.family));
    return agg;
  },

  coverage(specs, aggregate) {
    const coverage = new Map((aggregate?.capabilities || []).map(entry => [entry.capability, entry]));
    return (Array.isArray(specs) ? specs : [])
      .filter(spec => this.shouldQueryCapability(spec, 'snapshot'))
      .map(spec => {
        const id = spec.id || spec.capability_id;
        const entry = coverage.get(id);
        return {
          id,
          name: spec.name || id,
          readiness: spec.readiness_level || spec.readiness || '',
          family: this.category(id),
          queried: entry?.queried || 0,
          available: entry?.available || 0,
          missing: entry?.missing || 0,
          reason: entry?.reason || spec.summary || '',
        };
      });
  },

  selectMetrics(metrics, count = 4) {
    const source = metrics && typeof metrics === 'object' ? metrics : {};
    const aliases = {
      instruction_count: 'instructions', function_count: 'functions',
      remark_count: 'remarks', symbol_count: 'symbols', global_count: 'globals',
    };
    const merged = {};
    for (const [key, value] of Object.entries(source)) {
      if (typeof value !== 'number' || !Number.isFinite(value)) continue;
      const canonical = aliases[key] || key;
      merged[canonical] = (merged[canonical] || 0) + value;
    }
    const chosen = [];
    const seen = new Set();
    const canonicalOrder = [...new Set(preferredMetricOrder.map(k => aliases[k] || k))];
    canonicalOrder.forEach(key => {
      const value = merged[key];
      if (value == null) return;
      if (value === 0 && key !== 'errors' && key !== 'warnings') return;
      chosen.push({ key, label: titleCase(key), value });
      seen.add(key);
    });
    Object.entries(merged)
      .filter(([key, value]) => !seen.has(key) && typeof value === 'number' && Number.isFinite(value))
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
      .slice(0, Math.max(0, count - chosen.length))
      .forEach(([key, value]) => chosen.push({ key, label: titleCase(key), value }));
    return chosen.slice(0, count);
  },

  chartRows(result) {
    const value = result?.value || {};
    if (value.by_type && typeof value.by_type === 'object') {
      return Object.entries(value.by_type)
        .filter(([, raw]) => typeof raw === 'number' && Number.isFinite(raw))
        .map(([label, amount]) => ({ label, amount }));
    }
    const table = (result?.tables || []).find(t =>
      Array.isArray(t.rows) &&
      t.rows.length > 1 &&
      t.rows.length <= 16 &&
      t.rows.every(row => row && typeof row === 'object')
    );
    if (!table) return [];
    const sample = table.rows[0];
    const labelKey = ['name', 'type', 'kind', 'pass', 'function', 'section'].find(key => key in sample);
    const valueKey = Object.keys(sample).find(key => typeof sample[key] === 'number');
    if (!labelKey || !valueKey) return [];
    return table.rows
      .filter(row => typeof row[valueKey] === 'number')
      .map(row => ({ label: row[labelKey], amount: row[valueKey] }));
  },
};

// --- UI Components ---
const UI = {
  metric(label, value, subtext, cls) {
    return html`<div class="metric-card">
      <div class="label">${label}</div>
      <div class="value">${typeof value === 'number' ? formatNumber(value) : value}</div>
      ${subtext ? html`<div class="delta ${cls || 'neutral'}">${subtext}</div>` : null}
    </div>`;
  },

  dataTable(rows, options = {}) {
    const list = Array.isArray(rows) ? rows : [];
    if (!list.length) return html`<div class="empty-state"><div>No rows</div></div>`;
    const columns = options.columns || Object.keys(list[0]).slice(0, 8);
    return h('div', { class: 'data-table-wrap' },
      h('table', { class: 'data-table' },
        h('thead', {}, h('tr', {}, columns.map(c => h('th', {}, titleCase(c))))),
        h('tbody', {}, list.slice(0, options.limit || 200).map(row =>
          h('tr', {}, columns.map(c => h('td', { class: typeof row[c] === 'number' ? 'mono number' : '' }, row[c] == null ? '–' : String(row[c]))))
        ))
      )
    );
  },

  findingList(findings) {
    const list = Array.isArray(findings) ? findings : [];
    if (!list.length) return html`<div class="empty-state"><div>No findings</div></div>`;
    return h('div', { class: 'finding-list' }, list.slice(0, 300).map(f => {
      const sev = String(f.severity || f.level || 'info').toLowerCase();
      const loc = [f.file, f.line, f.column].filter(v => v != null && v !== '').join(':');
      return h('div', { class: `finding-row ${sev}` },
        h('span', { class: `severity-badge ${sev}` }, sev),
        h('div', { class: 'finding-body' },
          h('div', { class: 'finding-message' }, f.message || f.name || '(no message)'),
          h('div', { class: 'finding-meta mono' }, [f.pass, f.function, loc, f.hotness != null ? `hotness ${f.hotness}` : ''].filter(Boolean).join(' · '))
        )
      );
    }));
  },

  barChart(rows, options = {}) {
    const list = Array.isArray(rows) ? rows : [];
    if (!list.length) return null;
    const max = Math.max(...list.map(item => Number(item.amount) || 0), 1);
    const colors = options.colors || ['#5B8DB8', '#5DB8A8', '#D4A574', '#9DB86E', '#C97DB8', '#9B7DB8', '#D48B9B', '#6EC9C4', '#F0E8B8', '#D6C5E8', '#E8D6B8', '#E8B8C5'];
    return h('div', { class: 'bar-chart' }, list.slice(0, options.limit || 12).map((item, i) =>
      h('div', { class: 'bar-row' },
        h('div', { class: 'bar-label', title: item.label }, item.label),
        h('div', { class: 'bar-track' },
          h('div', {
            class: 'bar-fill',
            style: { width: `${(Number(item.amount) || 0) === 0 ? 0 : Math.max(6, Math.round((Number(item.amount) || 0) / max * 100))}%`, background: item.color || colors[i % colors.length] }
          })
        ),
        h('div', { class: 'bar-value mono' }, formatNumber(item.amount))
      )
    ));
  },

  passTimeline(remarks) {
    const list = (Array.isArray(remarks) ? remarks : []).filter(r => r.pass);
    if (!list.length) return null;
    const byPass = new Map();
    list.forEach(r => {
      const p = r.pass;
      if (!byPass.has(p)) byPass.set(p, { pass: p, count: 0, hotness: 0 });
      byPass.get(p).count++;
      byPass.get(p).hotness += Number(r.hotness || 0);
    });
    const entries = Array.from(byPass.values()).sort((a, b) => b.count - a.count);
    const max = Math.max(...entries.map(e => e.count), 1);
    const colors = ['#5B8DB8', '#5DB8A8', '#D4A574', '#9DB86E', '#C97DB8', '#9B7DB8', '#D48B9B'];
    return h('div', { class: 'bar-chart' }, entries.slice(0, 10).map((e, i) =>
      h('div', { class: 'bar-row' },
        h('div', { class: 'bar-label', title: e.pass }, e.pass),
        h('div', { class: 'bar-track' },
          h('div', {
            class: 'bar-fill',
            style: { width: `${Math.max(6, Math.round(e.count / max * 100))}%`, background: colors[i % colors.length] }
          })
        ),
        h('div', { class: 'bar-value mono' }, `${e.count}${e.hotness ? ' · h:' + formatNumber(e.hotness) : ''}`)
      )
    ));
  },

  flameBars(items, options = {}) {
    const list = Array.isArray(items) ? items : [];
    if (!list.length) return null;
    const total = list.reduce((s, it) => s + (Number(it.value) || 0), 0) || 1;
    const colors = ['#5B8DB8', '#5DB8A8', '#D4A574', '#9DB86E', '#C97DB8', '#9B7DB8', '#D48B9B', '#6EC9C4'];
    return h('div', { class: 'flame-bar', style: { gap: '2px', margin: '8px 0' } },
      list.slice(0, options.limit || 20).map((it, i) => {
        const pct = ((Number(it.value) || 0) / total) * 100;
        return h('div', {
          class: 'flame-bar-segment',
          style: { width: `${pct}%`, background: it.color || colors[i % colors.length] }
        }, h('span', { title: `${it.label}: ${formatNumber(it.value)} (${pct.toFixed(1)}%)` }, it.label));
      })
    );
  },

  errorCard(message, onRetry) {
    return h('div', { class: 'empty-state', style: { padding: '24px' } },
      h('div', {}, 'Something went wrong'),
      h('div', { class: 'reason mono' }, message || 'Unknown error'),
      onRetry ? h('button', { class: 'retry-btn', onClick: onRetry }, 'Retry') : null
    );
  },

  capabilityPanel(result) {
    const nodes = [];
    if (!result.available)
      nodes.push(html`<div class="empty-state"><div>${result.reason || 'Capability unavailable'}</div></div>`);
    const metricEntries = Object.entries(result.metrics || {});
    if (metricEntries.length) {
      nodes.push(h('div', { class: 'mini-metrics' }, metricEntries.slice(0, 12).map(([k, v]) =>
        h('div', { class: 'mini-metric' }, h('span', {}, titleCase(k)), h('strong', { class: 'mono' }, typeof v === 'number' ? formatNumber(v) : String(v)))
      )));
    }
    const chart = CapabilityData.chartRows(result);
    if (chart.length)
      nodes.push(this.barChart(chart));
    if (result.findings.length) nodes.push(this.findingList(result.findings));
    result.tables.slice(0, 2).forEach(t => nodes.push(html`<div class="table-title">${titleCase(t.name)}</div>`, this.dataTable(t.rows)));
    if (!nodes.length) {
      const raw = result.value || {};
      const rawEntries = Object.entries(raw).filter(([k, v]) => !ignoredMetricKeys.has(k) && v != null && typeof v !== 'object');
      if (rawEntries.length) {
        nodes.push(h('div', { class: 'mini-metrics' }, rawEntries.slice(0, 16).map(([k, v]) =>
          h('div', { class: 'mini-metric' }, h('span', {}, titleCase(k)), h('strong', { class: 'mono' }, typeof v === 'number' ? formatNumber(v) : String(v)))
        )));
      } else {
        nodes.push(html`<div class="empty-state"><div>No data to display for this capability.</div></div>`);
      }
    }
    return h('div', { class: 'capability-panel' }, nodes);
  },

  donutChart(data, options = {}) {
    const list = Array.isArray(data) ? data : [];
    if (!list.length) return null;
    const total = list.reduce((s, d) => s + (Number(d.value) || 0), 0) || 1;
    const colors = options.colors || ['#5B8DB8', '#5DB8A8', '#D4A574', '#9DB86E', '#C97DB8', '#9B7DB8', '#D48B9B'];
    const size = options.size || 120;
    const r = (size - 8) / 2;
    const cx = size / 2;
    const cy = size / 2;
    const circ = 2 * Math.PI * r;
    const svgNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('width', String(size));
    svg.setAttribute('height', String(size));
    svg.setAttribute('viewBox', `0 0 ${size} ${size}`);

    let offset = -circ / 4;
    list.forEach((d, i) => {
      const val = Number(d.value) || 0;
      const pct = val / total;
      const dash = pct * circ;
      const circle = document.createElementNS(svgNS, 'circle');
      circle.setAttribute('cx', cx);
      circle.setAttribute('cy', cy);
      circle.setAttribute('r', r);
      circle.setAttribute('fill', 'none');
      circle.setAttribute('stroke', colors[i % colors.length]);
      circle.setAttribute('stroke-width', '10');
      circle.setAttribute('stroke-dasharray', `${dash.toFixed(2)} ${(circ - dash).toFixed(2)}`);
      circle.setAttribute('stroke-dashoffset', offset.toFixed(2));
      circle.setAttribute('stroke-linecap', 'round');
      circle.setAttribute('transform', `rotate(-90 ${cx} ${cy})`);
      svg.appendChild(circle);
      offset -= dash;
    });

    const legend = h('div', { class: 'overview-donut-legend' });
    list.forEach((d, i) => {
      const pct = total ? Math.round((Number(d.value) || 0) / total * 100) : 0;
      legend.appendChild(h('span', {},
        h('i', { style: { background: colors[i % colors.length] } }),
        `${d.label} · ${pct}%`
      ));
    });

    return h('div', { class: 'overview-donut' }, svg, legend);
  },

  sparkline(data, options = {}) {
    const list = Array.isArray(data) ? data : [];
    if (list.length < 2) return null;
    const w = options.width || 280;
    const h = options.height || 60;
    const pad = 4;
    const svgNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('width', String(w));
    svg.setAttribute('height', String(h));
    svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
    const color = options.color || 'var(--accent)';

    const values = list.map(d => Number(d.value) || 0);
    const max = Math.max(...values, 1);
    const min = Math.min(...values, 0);
    const range = max - min || 1;
    const step = (w - pad * 2) / (list.length - 1);

    const points = list.map((_, i) => {
      const x = pad + i * step;
      const y = h - pad - ((values[i] - min) / range) * (h - pad * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });

    const areaPoints = `${points[0].split(',')[0]},${h - pad} ` + points.join(' ') + ` ${points[points.length - 1].split(',')[0]},${h - pad}`;
    const area = document.createElementNS(svgNS, 'polygon');
    area.setAttribute('points', areaPoints);
    area.setAttribute('fill', color);
    area.setAttribute('opacity', '0.12');
    svg.appendChild(area);

    const poly = document.createElementNS(svgNS, 'polyline');
    poly.setAttribute('points', points.join(' '));
    poly.setAttribute('fill', 'none');
    poly.setAttribute('stroke', color);
    poly.setAttribute('stroke-width', '2');
    poly.setAttribute('stroke-linejoin', 'round');
    poly.setAttribute('stroke-linecap', 'round');
    svg.appendChild(poly);

    list.forEach((_, i) => {
      const [x, y] = points[i].split(',');
      const circle = document.createElementNS(svgNS, 'circle');
      circle.setAttribute('cx', x);
      circle.setAttribute('cy', y);
      circle.setAttribute('r', '2.5');
      circle.setAttribute('fill', color);
      svg.appendChild(circle);
    });

    return svg;
  },

  radarChart(axes, options = {}) {
    const list = Array.isArray(axes) ? axes.filter(a => a.label) : [];
    if (list.length < 3) return null;
    const size = options.size || 200;
    const cx = size / 2, cy = size / 2;
    const radius = (size - 60) / 2;
    const color = options.color || 'var(--accent)';
    const svgNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('width', String(size));
    svg.setAttribute('height', String(size));
    svg.setAttribute('viewBox', `0 0 ${size} ${size}`);

    const n = list.length;
    const angleStep = (2 * Math.PI) / n;
    const vertex = (i, r) => {
      const a = -Math.PI / 2 + i * angleStep;
      return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
    };

    [0.33, 0.66, 1].forEach(pct => {
      const pts = Array.from({ length: n }, (_, i) => vertex(i, radius * pct).map(v => v.toFixed(1)).join(',')).join(' ');
      const poly = document.createElementNS(svgNS, 'polygon');
      poly.setAttribute('points', pts);
      poly.setAttribute('fill', 'none');
      poly.setAttribute('stroke', 'var(--border)');
      poly.setAttribute('stroke-width', '1');
      svg.appendChild(poly);
    });

    list.forEach((_, i) => {
      const [x, y] = vertex(i, radius);
      const line = document.createElementNS(svgNS, 'line');
      line.setAttribute('x1', cx); line.setAttribute('y1', cy);
      line.setAttribute('x2', x.toFixed(1)); line.setAttribute('y2', y.toFixed(1));
      line.setAttribute('stroke', 'var(--border)');
      line.setAttribute('stroke-width', '1');
      svg.appendChild(line);
    });

    const dataPts = list.map((a, i) => {
      const pct = a.max > 0 ? Math.min(a.value / a.max, 1) : 0;
      return vertex(i, radius * pct).map(v => v.toFixed(1)).join(',');
    }).join(' ');
    const dataPoly = document.createElementNS(svgNS, 'polygon');
    dataPoly.setAttribute('points', dataPts);
    dataPoly.setAttribute('fill', color);
    dataPoly.setAttribute('fill-opacity', '0.15');
    dataPoly.setAttribute('stroke', color);
    dataPoly.setAttribute('stroke-width', '2');
    svg.appendChild(dataPoly);

    list.forEach((a, i) => {
      const [x, y] = vertex(i, radius + 16);
      const text = document.createElementNS(svgNS, 'text');
      text.setAttribute('x', x.toFixed(1));
      text.setAttribute('y', y.toFixed(1));
      text.setAttribute('text-anchor', x < cx - 5 ? 'end' : x > cx + 5 ? 'start' : 'middle');
      text.setAttribute('dominant-baseline', y < cy - 5 ? 'auto' : y > cy + 5 ? 'hanging' : 'middle');
      text.setAttribute('font-size', '10');
      text.setAttribute('fill', 'var(--fg2)');
      text.setAttribute('font-family', 'var(--mono)');
      text.textContent = a.label;
      svg.appendChild(text);
    });

    return h('div', { class: 'radar-chart' }, svg);
  },

  deltaBar(items, options = {}) {
    const list = Array.isArray(items) ? items : [];
    if (!list.length) return null;
    const maxAbs = Math.max(...list.map(it => Math.abs(it.delta ?? (it.after - it.before))), 1);
    return h('div', { class: 'delta-bar' }, list.slice(0, options.limit || 12).map(item => {
      const delta = item.delta ?? ((item.after || 0) - (item.before || 0));
      const pct = Math.min(Math.abs(delta) / maxAbs * 50, 50);
      const cls = delta > 0 ? 'positive' : delta < 0 ? 'negative' : 'neutral';
      const sign = delta > 0 ? '+' : '';
      return h('div', { class: 'delta-row' },
        h('div', { class: 'delta-label' }, item.label || ''),
        h('div', { class: 'delta-track' },
          h('div', { class: 'delta-center' }),
          delta !== 0 ? h('div', {
            class: `delta-fill ${cls}`,
            style: { width: `${pct}%`, ...(delta < 0 ? { right: '50%' } : { left: '50%' }) }
          }) : null
        ),
        h('div', { class: `delta-value ${cls}` }, delta === 0 ? '0' : `${sign}${formatNumber(delta)}`)
      );
    }));
  },

  heatmapGrid(rows, cols, getData, options = {}) {
    if (!rows.length || !cols.length) return null;
    const cellSize = options.cellSize || 24;
    const labelW = 100, labelH = 60;
    const w = labelW + cols.length * cellSize;
    const ht = labelH + rows.length * cellSize;
    const svgNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('width', String(Math.min(w, 600)));
    svg.setAttribute('viewBox', `0 0 ${w} ${ht}`);
    svg.style.maxWidth = '100%';

    cols.forEach((col, ci) => {
      const text = document.createElementNS(svgNS, 'text');
      const x = labelW + ci * cellSize + cellSize / 2;
      text.setAttribute('x', String(x));
      text.setAttribute('y', String(labelH - 6));
      text.setAttribute('text-anchor', 'end');
      text.setAttribute('transform', `rotate(-45 ${x} ${labelH - 6})`);
      text.setAttribute('font-size', '9');
      text.setAttribute('fill', 'var(--fg3)');
      text.setAttribute('font-family', 'var(--mono)');
      text.textContent = col.length > 12 ? col.slice(0, 11) + '…' : col;
      svg.appendChild(text);
    });

    rows.forEach((row, ri) => {
      const text = document.createElementNS(svgNS, 'text');
      text.setAttribute('x', String(labelW - 6));
      text.setAttribute('y', String(labelH + ri * cellSize + cellSize / 2 + 3));
      text.setAttribute('text-anchor', 'end');
      text.setAttribute('font-size', '9');
      text.setAttribute('fill', 'var(--fg3)');
      text.setAttribute('font-family', 'var(--mono)');
      text.textContent = row.length > 14 ? row.slice(0, 13) + '…' : row;
      svg.appendChild(text);

      cols.forEach((col, ci) => {
        const val = getData(ri, ci);
        const rect = document.createElementNS(svgNS, 'rect');
        rect.setAttribute('x', String(labelW + ci * cellSize + 1));
        rect.setAttribute('y', String(labelH + ri * cellSize + 1));
        rect.setAttribute('width', String(cellSize - 2));
        rect.setAttribute('height', String(cellSize - 2));
        rect.setAttribute('rx', '3');
        rect.setAttribute('fill', val > 0 ? `rgba(48,209,198,${Math.min(val, 1) * 0.7 + 0.15})` : 'var(--bg3)');
        rect.setAttribute('class', 'heatmap-cell');
        const title = document.createElementNS(svgNS, 'title');
        title.textContent = `${row} × ${col}: ${val > 0 ? 'available' : 'missing'}`;
        rect.appendChild(title);
        svg.appendChild(rect);
      });
    });

    return h('div', { class: 'heatmap-grid' }, svg);
  },

  inspectResult(result) {
    if (!result || typeof result !== 'object')
      return html`<div class="empty-state"><div>No inspection result</div></div>`;

    if (result.baseline && result.candidate) {
      const baseline = CapabilityData.normalizeResults([{ capability: result.capability, value: result.baseline.value }])[0];
      const candidate = CapabilityData.normalizeResults([{ capability: result.capability, value: result.candidate.value }])[0];
      return h('div', { class: 'capability-stack' },
        h('section', { class: 'capability-card' },
          h('div', { class: 'capability-card-title mono' }, `Baseline · ${(result.baseline_snapshot_id || '').slice(0, 8)}`),
          this.capabilityPanel(baseline)
        ),
        h('section', { class: 'capability-card' },
          h('div', { class: 'capability-card-title mono' }, `Candidate · ${(result.snapshot_id || '').slice(0, 8)}`),
          this.capabilityPanel(candidate)
        ),
        h('section', { class: 'capability-card' },
          h('div', { class: 'capability-card-title mono' }, 'Diff'),
          this.capabilityPanel({
            capability: result.capability,
            available: true,
            value: result.diff || {},
            metrics: CapabilityData.metrics(result.diff || {}),
            tables: CapabilityData.tables(result.diff || {}),
            findings: CapabilityData.findings(result.diff || {}),
            artifacts: CapabilityData.artifacts(result.diff || {}),
          })
        )
      );
    }

    if (Array.isArray(result.signals)) {
      const normalized = result.signals.map(entry =>
        CapabilityData.normalizeResults([{ capability: entry.capability, value: entry.value }])[0]
      );
      return h('div', { class: 'capability-stack' }, normalized.map(entry =>
        h('section', { class: 'capability-card' },
          h('div', { class: 'capability-card-title mono' }, friendlyCapabilityName(entry.capability)),
          this.capabilityPanel(entry)
        )
      ));
    }

    if (result.value) {
      const normalized = CapabilityData.normalizeResults([{ capability: result.capability, value: result.value }])[0];
      return h('div', { class: 'capability-stack' },
        h('section', { class: 'capability-card' },
          h('div', { class: 'capability-card-title mono' }, `${friendlyCapabilityName(result.capability)} · ${result.source_path || result.unit_selector || ''}`.trim()),
          this.capabilityPanel(normalized)
        )
      );
    }

    return html`<div class="empty-state"><div>No data available</div></div>`;
  },
};
